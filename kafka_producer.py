from kafka import KafkaProducer
import json
import time

KAFKA_BROKER = "localhost:9092"
TOPIC = "drone-images-topic"

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

print("Kafka Producer Started...")

while True:
    message = input("Enter message: ")

    data = {
        "message": message
    }

    producer.send(TOPIC, value=data)
    producer.flush()

    print("Sent:", data)