import time
import logging


def get_logger(logDir):
    nowTime = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    logPath = f"{logDir}/{nowTime}.log"
    
    logger = logging.getLogger(logPath)
    logger.setLevel(logging.DEBUG)
    
    if logger.root.handlers:
        logger.root.handlers[0].setLevel(logging.WARNING)
    
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    
    fh = logging.FileHandler(logPath)
    sh = logging.StreamHandler()
    fh.setLevel(logging.INFO)
    sh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(sh)
    
    return logger


def log_loss(logger, record, epoch):
    logger.info(f'Epoch: {epoch:>3d};' +
                f' D loss: {record.D:>7.5f};'       +
                f' G loss: {record.G:>7.5f};'       +
                f' GAdv loss: {record.GAdv:>7.5f};' +
                f' GCmp loss: {record.GCmp:>7.5f};' +
                f' GPer loss: {record.GPer:>7.5f};' + '\n')


def write_loss(writer, record, tag2, step):
    writer.add_scalar(f'D_loss/{tag2}', record.D, step)
    writer.add_scalar(f'G_loss/{tag2}', record.G, step)
    writer.add_scalar(f'GAdv_loss/{tag2}', record.GAdv, step)
    writer.add_scalar(f'GCmp_loss/{tag2}', record.GCmp, step)
    writer.add_scalar(f'GPer_loss/{tag2}', record.GPer, step)


def log_and_write(logger, writer, record, tag2, step):
    pass
