Multithreading Assignment

 # Overview
This project demonstrates two multithreading problems:

1. **Producer-Consumer Problem** using Python Threads (`threading.Condition`)
2. **Matrix Multiplication** using Python Threads + PyTorch with Animation

#  Files in this Repository
| File | Description |
| `producer_consumer.py` | Producer-Consumer problem using Python threads and condition variables |
| `matrix_multiplication.py` | 100×100 matrix multiplication using Threads + PyTorch with animation |
| `matrix_animation.gif` | GIF demonstration of the matrix multiplication |
| `README.md` | Project documentation |


 1️⃣ Producer-Consumer Problem

The Producer-Consumer problem is implemented using Python threads and a shared, fixed-size bounded buffer.

- Two **producer** threads add items to the buffer.
- Two **consumer** threads remove items from the buffer.
- A single `threading.Lock` protects the shared buffer.
- `threading.Condition` objects (`not_full`, `not_empty`) are used to block/wake threads instead of busy-waiting.
- `not_full.wait()` — producers wait here when the buffer is full.
- `not_empty.wait(timeout=0.5)` — consumers wait here when the buffer is empty, and periodically re-check for shutdown.
- `notify()` wakes exactly one waiting thread on the opposite condition after every produce/consume.
- A `threading.Event` (`producers_done`) signals consumers to shut down cleanly once all producers have finished and the buffer is empty.

The buffer size is 5, with 2 producers each generating 10 items (20 items total).

#  Concepts Used
- Python Threads (`threading.Thread`)
- Shared Resource (bounded list buffer)
- Synchronization (`threading.Lock`)
- `threading.Condition` — `wait()` / `notify()`
- Graceful shutdown with `threading.Event`

### ▶️ How to Run

python producer_consumer.py

# Sample Output
```
Producer 0 produced: item-0-0 | Buffer size: 1
Producer 1 produced: item-1-0 | Buffer size: 2
Consumer 0 consumed: item-0-0 | Buffer size: 1
Producer 0 produced: item-0-1 | Buffer size: 2
Consumer 1 consumed: item-1-0 | Buffer size: 1
...
All producers and consumers have finished.
```

---

 2️⃣ Matrix Multiplication using Threads + PyTorch

Two 100×100 matrices are multiplied using Python threads and PyTorch tensors, with the rows of matrix A split evenly across **4 worker threads**.

#  How It Works
- `ThreadPoolExecutor` creates a pool of 4 worker threads.
- Each thread is assigned a contiguous block of rows of `A`.
- For each row, PyTorch computes `torch.matmul(row_A, B)` to produce that row of the result.
- A `threading.Lock` protects writes to the shared result matrix `C` and the `completed_rows` progress tracker.
- Matplotlib's `FuncAnimation` polls the shared state every frame and reveals the animation live, in the actual order rows are completed by the threads.
- After computation, the threaded result is verified against `torch.matmul(A, B)` using `torch.allclose()`.

# Animation
The animation shows:

- 🔵 **Matrix A** – a moving red horizontal line indicates the row currently being processed.
- 🟢 **Matrix B** – a moving red vertical line sweeps across columns.
- 🟠 **Matrix C** – the result matrix fills in live, row by row, as threads complete their work.

# Animation Output


### ▶️ Installation
Install the required packages:pip install torch matplotlib

# Run
python matrix_multiplication.py
The animation window opens immediately and updates live as the background threads compute each row. Once complete, the console prints the verification result and a preview of `A`, `B`, and `C`.

#  Sample Output

 MATRIX MULTIPLICATION STARTED
Matrix A : torch.Size([100, 100])
Matrix B : torch.Size([100, 100])
Matrix C : torch.Size([100, 100])
Threads  : 4
Tensor Framework : PyTorch

Thread-1 completed row 1
Thread-2 completed row 26
...
       ALL THREADS COMPLETED

             VERIFYING RESULT

RESULT VERIFIED SUCCESSFULLY!
Threaded result is correct.
PyTorch verification: PASSED
```

---

# Technologies Used
- Python
- PyTorch
- ThreadPoolExecutor
- Matplotlib (FuncAnimation)
- `threading` (Lock, Condition, Event)

#  Requirements
- Python 3.9 or above
- `torch`
- `matplotlib`

Install dependencies using:
pip install torch matplotlib


