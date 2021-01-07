DEFAULT_TB_GATEWAY_CONF = "/etc/thingsboard-gateway/config/tb_gateway.yaml"
DEFAULT_NDU_GATE_CONF = "/etc/ndu-gate/config/ndu_gate.yaml"
DEFAULT_NDU_GATE_CONF_WIN = "C:/ndu-gate/config/ndu_gate.yaml"

NDU_GATE_MODEL_FOLDER = "/etc/ndu-gate/data"

DEFAULT_HANDLER_SETTINGS = {
    "type": "SOCKET",
    "socket": {
        "port": 60060,
        "host": "127.0.0.1"
    }
}


RESULT_KEY_TRACK_ID = "track_id"
# [x1, y1, x2, y2]
RESULT_KEY_RECT = "rect"
RESULT_KEY_RECT_COLOR = "rect_color"
RESULT_KEY_RECT_DEBUG_TEXT = "rect_debug_text"
# string
RESULT_KEY_CLASS_NAME = "class_name"
# string - used for debug previewing
RESULT_KEY_PREVIEW_KEY = "preview_key"
# 0-1 arasında float olasılık değeri.
RESULT_KEY_SCORE = "score"
# Platform'a gönderilmesi istenen veri. İçeriği dictionary olmalıdır.
RESULT_KEY_DATA = "data"
# Preview'e büyük şekilde yazılacak veri.
RESULT_KEY_DEBUG = "debug"

EXTRA_DATA_KEY_RESULTS = "results"
