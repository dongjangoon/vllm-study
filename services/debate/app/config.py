import os

LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "http://litellm-proxy.litellm.svc.cluster.local:4000")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "<YOUR_MASTER_KEY>")
LITELLM_MODEL = os.getenv("LITELLM_MODEL", "mistral-7b-awq")

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-placeholder")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-placeholder")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "http://langfuse-web.langfuse.svc.cluster.local:3000")
