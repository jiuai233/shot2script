"""按余弦相似度做平均连接凝聚聚类，不预设人数。"""

import numpy as np

from .embeddings import unit


def agglomerate(vectors, threshold):
    if not vectors:
        return []
    matrix = np.clip(np.asarray(vectors) @ np.asarray(vectors).T, -1, 1)
    groups = [[i] for i in range(len(vectors))]
    while len(groups) > 1:
        best, pair = -2.0, None
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                score = float(matrix[np.ix_(groups[i], groups[j])].mean())
                if score > best:
                    best, pair = score, (i, j)
        if best < threshold:
            break
        i, j = pair
        groups[i].extend(groups.pop(j))
    return sorted((sorted(group) for group in groups), key=lambda g: g[0])


def assign(vectors, eligible, threshold):
    indices = [i for i, (v, ok) in enumerate(zip(vectors, eligible)) if v is not None and ok]
    groups = [[indices[i] for i in group] for group in
              agglomerate([vectors[i] for i in indices], threshold)]
    centroids = [unit(np.mean([vectors[i] for i in group], axis=0)) for group in groups]
    labels = {i: label for label, group in enumerate(groups) for i in group}
    scores = []
    for i, vector in enumerate(vectors):
        if vector is None or not centroids:
            scores.append(None)
            continue
        similarities = np.clip(np.asarray(centroids) @ vector, -1, 1)
        label = labels.get(i)
        if label is None:
            candidate = int(np.argmax(similarities))
            if similarities[candidate] > threshold:
                labels[i] = label = candidate
        scores.append(float(similarities[label]) if label is not None else float(similarities.max()))
    matrix = np.clip(np.asarray(centroids) @ np.asarray(centroids).T, -1, 1).tolist() if centroids else []
    return labels, scores, matrix
