from kafka import KafkaConsumer
import json

KAFKA_BROKER = "localhost:9092"
TOPIC = "drone-images-topic"

def safe_deserializer(m):
    try:
        return json.loads(m.decode("utf-8"))
    except:
        return m.decode("utf-8")

consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers=KAFKA_BROKER,
    auto_offset_reset="latest",
    group_id="drone-group",
    value_deserializer=safe_deserializer
)

print("Kafka Consumer Started...")

for message in consumer:
    print("Received:", message.value)