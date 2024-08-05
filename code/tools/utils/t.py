import torch


def cosine_distance_matrix_with_mean(matrix):
    # 归一化矩阵的每一行
    norm_matrix = matrix / matrix.norm(dim=1, keepdim=True)

    # 计算余弦相似度矩阵
    cosine_similarity_matrix = torch.mm(norm_matrix, norm_matrix.t())

    # 将余弦相似度转换为余弦距离
    cosine_distance_matrix = 1 - cosine_similarity_matrix

    # 计算每一行的均值，忽略对角线上的0值
    row_means = cosine_distance_matrix.sum(dim=1) / (cosine_distance_matrix.size(1) - 1)

    return cosine_distance_matrix, row_means


# 示例矩阵
matrix = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0]
])

# 计算每一行和其他行的余弦距离矩阵和均值
distance_matrix, row_means = cosine_distance_matrix_with_mean(matrix)
print("Distance Matrix:\n", distance_matrix)
print("Row Means:\n", row_means.unsqueeze(1))
