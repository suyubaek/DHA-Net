import os
import json
import time
import hmac
import base64
import hashlib
import logging
from typing import Optional

try:
    # 优先使用标准库，避免额外依赖
    from urllib.request import Request, urlopen
    from urllib.error import URLError, HTTPError
except Exception:  # pragma: no cover
    Request = None
    urlopen = None
    URLError = Exception
    HTTPError = Exception


logger = logging.getLogger(__name__)


def _gen_sign(secret: str) -> tuple[str, str]:
    """
    生成飞书群机器人签名。
    按官方要求：sign = base64(HMAC-SHA256(secret, f"{timestamp}\n{secret}"))
    返回 (timestamp, sign)
    """
    ts = str(int(time.time()))
    string_to_sign = f"{ts}\n{secret}"
    digest = hmac.new(secret.encode("utf-8"), string_to_sign.encode("utf-8"), hashlib.sha256).digest()
    sign = base64.b64encode(digest).decode("utf-8")
    return ts, sign


def send_message(title: str, content: str, *, msg_type: str = "post") -> bool:
    """
    发送训练状态到飞书（Lark）。

    环境变量：
    - LARK_WEBHOOK_URL：飞书群机器人 Webhook 地址（必需）
    - LARK_SECRET：机器人安全设置的密钥（可选）

    参数：
    - title：消息标题
    - content：消息正文（支持多行字符串）
    - msg_type："post"（富文本）或 "text"（纯文本）

    返回：
    - True 表示发送成功；False 表示未配置或发送失败。
    """
    webhook = os.environ.get("LARK_WEBHOOK_URL", "").strip()
    secret = os.environ.get("LARK_SECRET", "").strip()

    if not webhook:
        logger.warning("LARK_WEBHOOK_URL 未配置，跳过发送飞书消息。")
        return False

    # 构造 payload
    if msg_type == "text":
        payload = {
            "msg_type": "text",
            "content": {"text": f"{title}\n{content}"},
        }
    else:
        # post 富文本：带标题 + 正文
        payload = {
            "msg_type": "post",
            "content": {
                "post": {
                    "zh_cn": {
                        "title": title,
                        "content": [[{"tag": "text", "text": content}]],
                    }
                }
            },
        }

    # 可选签名
    if secret:
        ts, sign = _gen_sign(secret)
        payload.update({"timestamp": ts, "sign": sign})

    body = json.dumps(payload).encode("utf-8")
    req = Request(
        url=webhook,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(req, timeout=10) as resp:
            status = getattr(resp, "status", 200)
            if status == 200:
                return True
            logger.error(f"飞书消息发送失败，HTTP 状态码: {status}")
            return False
    except HTTPError as e:
        logger.error(f"飞书消息 HTTPError: {e}")
    except URLError as e:
        logger.error(f"飞书消息 URLError: {e}")
    except Exception as e:  # 捕获所有异常，避免训练崩溃
        logger.exception(f"飞书消息发送异常: {e}")

    return False


__all__ = ["send_message"]
