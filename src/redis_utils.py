import redis
import json
from typing import List, Dict


class RedisMemory:
    def __init__(self, session_id: str, host='localhost', port=6379, db=0):
        self.session_id = session_id
        self.key = f"chat:{session_id}"
        self.client = redis.Redis(host=host, port=port, db=db)

    def append(self, role: str, content: str):
        message = json.dumps({"role": role, "content": content})
        self.client.rpush(self.key, message)

    def get_last_n(self, n: int = 5) -> List[Dict[str, str]]:
        all_msgs = self.client.lrange(self.key, -n, -1)
        return [json.loads(msg) for msg in all_msgs]

    def get_all(self) -> List[Dict[str, str]]:
        all_msgs = self.client.lrange(self.key, 0, -1)
        return [json.loads(msg) for msg in all_msgs]
