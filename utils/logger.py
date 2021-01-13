import logging
import sys

# TODO create wrapper class for the Logger object
# Configure logging
logging.basicConfig(stream=sys.stdout, format='%(asctime)s [%(levelname)s] %(message)s')

# Logger singleton
logger = logging.getLogger('pytorch-boilerplate')
