from src.absa.inference import predict_and_aggregate

reviews = [
    {"content": "giao hàng nhanh, đóng gói kỹ, chất lượng tốt"},
    {"content": "hàng kém chất lượng, giao chậm"},
    {"content": "giá hợp lý, shop nhiệt tình"},
]

scores = predict_and_aggregate(reviews, model="logreg")
for asp, data in scores.items():
    print(f"{asp}: {data['pct']}% ({data['pos']} khen / {data['neg']} che)")
