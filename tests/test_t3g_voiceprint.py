import json
from pathlib import Path

from jsonschema import Draft202012Validator
import numpy as np
import pytest

from s2s.voiceprint.library import enroll, compare, read, save
from s2s.voiceprint.evaluation import evaluate, evaluate_listen

INFO = {"模型": "campp", "版本": "fixture-v1", "预处理": "fixture"}


def episode(root, name="E01", roles=("C001", "C002", "C003"), vectors=None):
    rows = [{"id": f"{name}-S{i+1:03d}", "起": f"00:0{i}.0", "止": f"00:0{i+1}.0", "文本": "台词。", "听辨": "清楚", "声线": f"{name}-V{i+1}"} for i in range(len(roles))]
    save(root/name/"声音记录.json", rows)
    save(root/name/"归属表.json", [{"句": r["id"], "说话人": role, "确定度": "确认"} for r,role in zip(rows, roles)])
    if vectors is None:
        vectors = list(np.eye(len(rows), dtype=np.float32))
    return rows, vectors, INFO


def schema(name, value):
    path = Path("tools/s2s/schemas") / (name + ".schema.json")
    Draft202012Validator(read(path)).validate(value)


def test_cross_episode_top3_no_vectors_and_permissions(tmp_path):
    first = episode(tmp_path)
    summary = enroll(tmp_path, "E01", extracted=first)
    assert "向量" not in json.dumps(summary, ensure_ascii=False)
    for path in (tmp_path/"声纹").glob("*.json"):
        schema("声纹库", read(path))
        assert path.stat().st_mode & 0o777 == 0o600
    second = episode(tmp_path, "E02", vectors=[np.array(v) for v in [[0,1,0],[0,0,1],[1,0,0]]])
    before = (tmp_path/"E02/归属表.json").read_bytes()
    result = compare(tmp_path,"E02", extracted=second)
    assert [r["候选"][0]["角色"] for r in result["声线"]] == ["C002", "C003", "C001"]
    assert all(len(r["候选"]) == 3 for r in result["声线"])
    assert "向量" not in json.dumps(result, ensure_ascii=False)
    assert (tmp_path/"E02/归属表.json").read_bytes() == before
    schema("声纹候选", result)


def test_idempotent_weighted_enrollment_and_role_reassignment(tmp_path):
    first = episode(tmp_path, roles=("C001", "C001"), vectors=[np.array([1.,0]), np.array([0.,1])])
    enroll(tmp_path, "E01", extracted=first)
    enroll(tmp_path, "E01", extracted=first)
    assert read(tmp_path/"声纹/C001.json")["句数"] == 2
    second = episode(tmp_path, "E02", roles=("C001",), vectors=[np.array([1.,0])])
    enroll(tmp_path,"E02",extracted=second)
    data = read(tmp_path/"声纹/C001.json")
    assert data["句数"] == 3
    np.testing.assert_allclose(data["向量"],[2/3,1/3])
    save(tmp_path/"E01/归属表.json", [])
    enroll(tmp_path,"E01",extracted=first)
    data = read(tmp_path/"声纹/C001.json")
    assert data["句数"] == 1 and data["来源集"] == ["E02"]


def test_unconfirmed_mixed_and_unknown_not_enrolled(tmp_path):
    extracted = episode(tmp_path)
    owners = read(tmp_path/"E01/归属表.json")
    owners[0]["确定度"] = "存疑"
    owners[1]["说话人"] = "E01-P01-T1"
    extracted[0][2]["听辨"] = "重叠"
    save(tmp_path/"E01/归属表.json",owners)
    result = enroll(tmp_path,"E01",extracted=extracted)
    assert result["角色"] == [] and len(result["跳过句"]) == 3


def test_confirmed_c_enrolls_while_null_n_and_t_are_skipped(tmp_path):
    extracted = episode(tmp_path, roles=('C001', None, 'E01-N001', 'E01-P01-T1'))
    owners = read(tmp_path / 'E01/归属表.json')
    owners[1]['确定度'] = '存疑'
    save(tmp_path / 'E01/归属表.json', owners)
    before = (tmp_path / 'E01/归属表.json').read_bytes()
    result = enroll(tmp_path, 'E01', extracted=extracted)
    assert result['角色'] == [{'角色': 'C001', '本集句数': 1}]
    assert result['跳过句'] == ['E01-S002', 'E01-S003', 'E01-S004']
    assert [p.name for p in (tmp_path / '声纹').glob('*.json')] == ['C001.json']
    assert read(tmp_path / '声纹/C001.json')['句数'] == 1
    assert (tmp_path / 'E01/归属表.json').read_bytes() == before


def test_model_mismatch_and_bad_dimensions_rejected(tmp_path):
    first = episode(tmp_path)
    enroll(tmp_path,"E01",extracted=first)
    second = episode(tmp_path,"E02")
    with pytest.raises(ValueError,match="版本"):
        compare(tmp_path,"E02",extracted=(*second[:2],{**INFO,"版本":"other"}))
    with pytest.raises(ValueError,match="维度"):
        compare(tmp_path,"E02",extracted=(second[0],[np.array([1.,0])]*3,INFO))
    with pytest.raises(ValueError,match="本集目录"):
        enroll(tmp_path,"E02",extracted=first)


def test_pending_collision_excluded_and_stale_rejected(tmp_path):
    first = episode(tmp_path)
    enroll(tmp_path,"E01",extracted=first)
    second = episode(tmp_path,"E02")
    pred = [{"句": r["id"], **{k:r[k] for k in ("起","止","文本","声线")}, "待定": i==0} for i,r in enumerate(second[0])]
    save(tmp_path/"E02/声纹聚类.json",{"句列表": pred})
    result = compare(tmp_path,"E02",extracted=second)
    assert result["声线"][0]["句数"] == 0 and result["声线"][0]["候选"] == []
    second[0][1]["文本"] = "改过的句子"
    with pytest.raises(ValueError,match="已变化"):
        compare(tmp_path,"E02",extracted=second)


def test_empty_library_has_no_invented_candidates(tmp_path):
    extracted = episode(tmp_path)
    result = compare(tmp_path,"E01",extracted=extracted)
    assert all(not r["候选"] for r in result["声线"])


def test_duplicate_and_missing_assignments_rejected(tmp_path):
    extracted = episode(tmp_path)
    owner = {"句":"E01-S001","说话人":"C001","确定度":"确认"}
    save(tmp_path/"E01/归属表.json",[owner,owner])
    with pytest.raises(ValueError,match="重复"):
        enroll(tmp_path,"E01",extracted=extracted)
    save(tmp_path/"E01/归属表.json",[{**owner,"句":"E01-S099"}])
    with pytest.raises(ValueError,match="之外"):
        enroll(tmp_path,"E01",extracted=extracted)


def test_evaluation_abstention_unmatched_and_purity():
    annotation = {"校对完成": True, "台词": [{"id":f"E01-S{i:03d}","说话人": role} for i,role in enumerate(["A","A","B","B"],1)]}
    result = {"句列表": [{"句": "E01-S001", "声线": "V1", "待定": False}, {"句": "E01-S002", "声线": "待定", "待定": True}, {"句": "E01-S003", "声线": "V1", "待定":False}]}
    score = evaluate(result,annotation,{"V1":"A"})
    assert score["覆盖率"] == .5 and score["已分配句纯度"] == .5
    assert score["全量多数类覆盖率"] == .25 and score["说话人准确率"] == .25
    assert score["说话人拆分类数"] == {"A":1,"B":1}
    with pytest.raises(ValueError,match="尚未"):
        evaluate(result,{**annotation,"校对完成": False})


def test_evaluation_independent_annotation_ids_and_missing_mapping():
    result = {"句列表": [{"句":"E01-S001","声线":"V1","待定":False,"起":"00:00.0","止":"00:01.0"}]}
    annotation = {"校对完成":True,"台词":[{"id":"L001","说话人":"H01","起":"00:00.0","止":"00:01.0"}]}
    assert evaluate(result,annotation,{"V1":"H01"})["说话人准确率"] == 1
    assert evaluate(result,annotation)["说话人准确率"] is None


def test_annotation_fixture_ready_for_human_truth():
    annotation = read("tests/fixtures/t3g/标注.json")
    predictions = [{"句": f"E01-S{i+1:03d}", "声线": "V1" if i != 1 else "V2", "待定": False,
                    "起": row["起"], "止": row["止"]} for i, row in enumerate(annotation["台词"])]
    result = evaluate({"句列表": predictions}, annotation, {"V1": "H01", "V2": "H02"})
    assert result["说话人准确率"] == 1 and result["已分配句纯度"] == 1


def test_listen_evaluation_partial_deletion_and_snapshot(tmp_path):
    rows,*_ = episode(tmp_path)
    decisions = [{"句号": rows[0]["id"],"决定":"主路","最终文本":"台词"}, {"句号":rows[1]["id"],"决定":"手改","最终文本":"别的台词"}, {"句号":rows[2]["id"],"决定":"删除","最终文本":""}]
    score = evaluate_listen(rows,{"声音快照":rows,"结果":decisions})
    assert score["已裁定非删除句一致率"] == .5 and score["删除句数"] == 1
    with pytest.raises(ValueError,match="快照"):
        evaluate_listen(rows,{"声音快照":[],"结果":decisions})
