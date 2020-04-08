from queue import Queue
from loguru import logger

class AutoQueue(Queue):
	def __init__(self, frame_limit=300):
		super().__init__(frame_limit)
		self.limit = frame_limit

	def put(self, frame):
		if self.full():
			# logger.debug('Full')
			self.get(block=False, timeout=1)
		super().put(frame, block=False, timeout=1)


def test():
	q = AutoQueue(10)
	logger.debug(q)
	logger.debug(q.qsize())
	for i in range(15):
		q.put(i)

	logger.debug(q.queue)
	logger.debug(q.qsize())


if __name__ == '__main__':
	test()