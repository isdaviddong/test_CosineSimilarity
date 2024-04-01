using MathNet.Numerics.LinearAlgebra;
using Azure.AI.OpenAI;

var apiKey = "👉openai_api_key";
var endpoint = new Uri("https://api.openai.com/v1/embeddings");
var client = new OpenAIClient(apiKey);

// 調用API並獲取嵌入向量
EmbeddingsOptions embeddingOptions = new EmbeddingsOptions()
{
    DeploymentName = "text-embedding-3-large",
    Input = { "老人" },
};

// 調用API並獲取嵌入向量
var returnValue = await client.GetEmbeddingsAsync(embeddingOptions);

// 創建一個新的列表來存儲嵌入向量
var embeddingVector1 = new List<double>();
foreach (var item in returnValue.Value.Data[0].Embedding.ToArray())
{
    embeddingVector1.Add(item);
}

// 調用API並獲取嵌入向量
embeddingOptions = new EmbeddingsOptions()
{
    DeploymentName = "text-embedding-3-large",
    Input = { "年長者" },
};

// 調用API並獲取嵌入向量
returnValue = await client.GetEmbeddingsAsync(embeddingOptions);

// 創建一個新的列表來存儲嵌入向量
var embeddingVector2 = new List<double>();
// 將嵌入向量轉換為數學向量
foreach (var item in returnValue.Value.Data[0].Embedding.ToArray())
{
    embeddingVector2.Add(item);
}

// 將嵌入向量轉換為數學向量
Vector<double> vector1 = Vector<double>.Build.DenseOfArray(embeddingVector1.ToArray());
Vector<double> vector2 = Vector<double>.Build.DenseOfArray(embeddingVector2.ToArray());

// 計算餘弦相似度
double similarity = CalculateCosineSimilarity(vector1, vector2);
Console.WriteLine($"餘弦相似度: {similarity}");

// 計算餘弦相似度
static double CalculateCosineSimilarity(Vector<double> vectorA, Vector<double> vectorB)
{
    // 確保兩個向量不是空的
    if (vectorA.Count == 0 || vectorB.Count == 0)
        throw new ArgumentException("Vectors must not be empty.");

    // 計算點積
    double dotProduct = vectorA.DotProduct(vectorB);
    // 計算向量A和向量B的模
    double magnitudeA = vectorA.L2Norm();
    double magnitudeB = vectorB.L2Norm();
    // 計算並返回餘弦相似度
    return dotProduct / (magnitudeA * magnitudeB);
}