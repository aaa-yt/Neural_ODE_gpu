from logging import StreamHandler, basicConfig, DEBUG, WARNING, getLogger, Formatter

mpl_logger = getLogger('matplotlib')
mpl_logger.setLevel(WARNING)

def setup_logger(log_file):
    format_str = '%(asctime)s@%(name)s %(levelname)s # %(message)s'
    basicConfig(filename=log_file, level=DEBUG, format=format_str)
    stream_handler = StreamHandler()
    stream_handler.setFormatter(Formatter(format_str))
    getLogger().addHandler(stream_handler)

if __name__ == '__main__':
    setup_logger("Neural_ODE/test/logs/test.log")
    logger = getLogger("test")
    logger.info("OK")