import importlib.util
import json
from pathlib import Path
import wave

import numpy as np

from s2s.diarize.integration import export_representatives


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_voiceprint_cli_evaluation(tmp_path, capsys):
    cli = module("t3g_voiceprint", "tools/voiceprint.py")
    annotation = {"校对完成":True,"台词":[{"id":"E01-S001","说话人":"H01"}]}
    result = {"句列表":[{"句":"E01-S001","待定":False,"声线":"E01-V1"}]}
    for name,data in [("result",result),("truth",annotation),("mapping",{"E01-V1":"H01"})]:
        (tmp_path/(name+".json")).write_text(json.dumps(data))
    assert cli.main(["评测",str(tmp_path/"result.json"),str(tmp_path/"truth.json"),"--声线角色映射",str(tmp_path/"mapping.json")]) == 0
    assert json.loads(capsys.readouterr().out)["说话人准确率"] == 1


def test_diarize_cli_constraint_options(tmp_path, monkeypatch, capsys):
    cli = module("t3g_diarize", "tools/diarize.py")
    path=tmp_path/"sounds.json";path.write_text("[]")
    def fake_run(source, records, **kwargs):
        assert kwargs["constraints"]["constraints"] == ["lipsync"]
        assert "max_speakers" not in kwargs["constraints"]
        return {"句列表":[],"类别":[],"运行":{}}
    monkeypatch.setattr(cli,"run",fake_run)
    assert cli.main(["missing.wav",str(path),str(tmp_path/"result.json"),"--约束","lipsync"]) == 0
    assert json.loads(capsys.readouterr().out)["类数"] == 0


def test_real_audio_representative_export(tmp_path):
    source=tmp_path/"source.wav"
    with wave.open(str(source),"wb") as writer:
        writer.setparams((1,2,16000,0,"NONE","not compressed"))
        writer.writeframes((np.sin(np.arange(16000)*.1)*3000).astype("<i2").tobytes())
    result={"类别":[{"声线":"E01-V1","代表句":[{"句":"E01-S001","文本":"台词","起":"00:00.0","止":"00:00.5"}]}]}
    index=export_representatives(source,result,tmp_path/"clips")
    assert len(index) == 1
    with wave.open(str(tmp_path/"clips"/index[0]["文件"]),"rb") as reader:
        assert reader.getnframes() == 8000
