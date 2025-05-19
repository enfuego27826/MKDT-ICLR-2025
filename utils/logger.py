import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self,log_dir):
        os.makedirs(log_dir,exist_ok=True)
        self.log_file = open(os.path.join(log_dir,"log.txt"),"a")
        self.tb_writer = SummaryWriter(log_dir)
    
    def log(self,msg,step=None,tag=None):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_msg = f"[{timestamp}] {msg}"
        print(full_msg)
        self.log_file.write(full_msg + "\n")
        self.log_file.flush()

        if step is not None and tag is not None:
            self.tb_writer.add_scalar(tag,float(msg.split()[-1]),step)
    
    def close(self):
        self.log_file.close()
        self.tb_writer.close()