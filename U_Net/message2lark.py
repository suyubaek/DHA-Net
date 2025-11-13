import requests
import json
import time
import hmac
import base64
import logging

FEISHU_WEBHOOK_URL = "https://open.feishu.cn/open-apis/bot/v2/hook/5cd99837-5be3-4438-975f-89697bb5250c"
FEISHU_SECRET = "wzn4LqIgfwN4TRk2Mecc1b" 

def gen_sign(secret, timestamp):
    """
    根据时间戳和密钥生成签名
    """
    # 拼接签名字符串
    string_to_sign = f'{timestamp}\n{secret}'
    
    # 使用 HMAC-SHA256 加密
    hmac_code = hmac.new(string_to_sign.encode("utf-8"), digestmod='sha256').digest()
    
    # Base64 编码
    sign = base64.b64encode(hmac_code).decode('utf-8')
    
    return sign


def send_message(title, content):
    """发送一个带有签名的消息到飞书。"""
    try:
        # 1. 获取当前时间戳
        timestamp = str(int(time.time()))
        
        # 2. 生成签名
        sign = gen_sign(FEISHU_SECRET, timestamp)

        # 3. 构造带有 timestamp 和 sign 的 payload
        payload = {
            "timestamp": timestamp,
            "sign": sign,
            "msg_type": "post",
            "content": {
                "post": {
                    "zh_cn": {
                        "title": title,
                        "content": [
                            [
                                {
                                    "tag": "text",
                                    "text": content
                                }
                            ]
                        ]
                    }
                }
            }
        }
        
        headers = {'Content-Type': 'application/json'}
        response = requests.post(FEISHU_WEBHOOK_URL, data=json.dumps(payload), headers=headers, timeout=10)
        
        logging.info(f"飞书 API 响应状态码: {response.status_code}")
        response_data = response.json()
        logging.info(f"飞书 API 响应内容: {response_data}")

        if response_data.get("StatusCode") == 0 or response_data.get("code") == 0:
            logging.info("飞书通知发送成功。")
        else:
            logging.error("飞书通知发送失败。")
            
    except Exception as e:
        logging.error(f"发送飞书通知时发生错误: {e}")