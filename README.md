api = curl --location 'http://localhost:8000/query/' \
--header 'Content-Type: application/json' \
--data '{
    "question":"give brief summary"
}'



server = uvicorn main:app --reload
