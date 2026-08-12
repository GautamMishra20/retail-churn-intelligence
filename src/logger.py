import logging
import os
from datetime import datetime

log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'logs')

os.makedirs(log_dir, exist_ok=True)

log_file = f"{datetime.now().strftime('%Y-%m-%d')}.log"
log_file_path = os.path.join(log_dir, log_file)

logging.basicConfig(
    filename=log_file_path,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

def get_logger(name):
    return logging.getLogger(name)