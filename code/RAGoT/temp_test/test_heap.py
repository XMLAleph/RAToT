import heapq
import copy

class ReasonState(object):
    """Tracks and records the state of a given search."""

    def __init__(self, json_data, command, score=0.0):
        """Keep track of different stages in the state

        :param json_data: some basic, json represntation of data
        """
        self._data = json_data
        self._score = score
        self._next = command

    def copy(self):
        """Does a deep copy of the state

        :returns: new search state
        """
        new_data = copy.deepcopy(self._data)
        new_score = copy.deepcopy(self._score)
        new_next = copy.deepcopy(self._next)

        return ReasonState(new_data, new_next, new_score)

    ## important to implement to work
    ## with the heap datastructures
    # 但是后面真的用了堆结构吗？？？感觉和堆没有关系？
    def __lt__(self, other):
        if self.score < other.score:
            return True
        return False

    def __eq__(self, other):
        if self.score == other.score:
            return True
        return False

    @property
    def data(self):
        return self._data

    @property
    def score(self):
        return self._score

    @property
    def next(self):
        return self._next

    @next.setter
    def next(self, value):
        self._next = value

    @data.setter
    def data(self, value):
        self._data = value

def plus(a):
    return a+1

def _plus(a):
    return a-1
    
state_1 = ReasonState(35, plus, 0)
state_2 = ReasonState(66, _plus, 0)
state_3 = ReasonState(89, _plus, 0)
heap = []
heapq.heappush(heap, state_1)
heapq.heappush(heap, state_2)
heapq.heappush(heap, state_3)
print(heap[0].data, heap[1].data)
current_state = heapq.heappop(heap)
print(current_state.data)
print(heap[0].data, heap[1].data)
'''
heap = []
heapq.heappush(heap, 2)
heapq.heappush(heap, 1)
print(heap)
current_state = heapq.heappop(heap)
print(heap)
'''

print("test end")