import threading
import time
import random


BUFFER_SIZE = 5
NUM_PRODUCERS = 2
NUM_CONSUMERS = 2
NUM_ITEMS_PER_PRODUCER = 10


buffer = []
lock = threading.Lock()
not_full = threading.Condition(lock)   
not_empty = threading.Condition(lock)  

def producer(pid):
    for i in range(NUM_ITEMS_PER_PRODUCER):
        item = f"item-{pid}-{i}"

        with not_full:
            
            while len(buffer) == BUFFER_SIZE:
                not_full.wait()

           
            buffer.append(item)
            print(f"Producer {pid} produced: {item} | Buffer size: {len(buffer)}")

           
            not_empty.notify()

       
        time.sleep(random.uniform(0.1, 0.3))

def consumer(cid):
    while True:
        with not_empty:
           
            while len(buffer) == 0:
               
                not_empty.wait(timeout=0.5)
              
                if len(buffer) == 0 and producers_done.is_set():
                    return

           
            item = buffer.pop(0)
            print(f"Consumer {cid} consumed: {item} | Buffer size: {len(buffer)}")

            
            not_full.notify()

        
        time.sleep(random.uniform(0.1, 0.3))


producers_done = threading.Event()

def main():
   
    producer_threads = [
        threading.Thread(target=producer, args=(i,), name=f"Producer-{i}")
        for i in range(NUM_PRODUCERS)
    ]

    consumer_threads = [
        threading.Thread(target=consumer, args=(i,), name=f"Consumer-{i}")
        for i in range(NUM_CONSUMERS)
    ]

    for t in producer_threads + consumer_threads:
        t.start()

    for t in producer_threads:
        t.join()

   
    producers_done.set()

   
    for t in consumer_threads:
        t.join()

    print("All producers and consumers have finished.")

if __name__ == "__main__":
    main()
