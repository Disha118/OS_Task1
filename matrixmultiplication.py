import torch
import threading
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from concurrent.futures import ThreadPoolExecutor




ROWS = 100
COLS = 100
NUM_THREADS = 4


ANIMATION_SPEED = 60




torch.manual_seed(10)

A = torch.randint(
    1,
    10,
    (ROWS, COLS),
    dtype=torch.float32
)

B = torch.randint(
    1,
    10,
    (ROWS, COLS),
    dtype=torch.float32
)



C = torch.zeros(
    (ROWS, COLS),
    dtype=torch.float32
)


completed_rows = torch.zeros(
    ROWS,
    dtype=torch.bool
)



thread_current_row = [-1] * NUM_THREADS

lock = threading.Lock()




def multiply_rows(start_row, end_row, thread_id):

    thread_name = "Thread-" + str(thread_id + 1)

    for i in range(start_row, end_row):

        

        with lock:
            thread_current_row[thread_id] = i


        

        row_A = A[i]


        

        result = torch.matmul(
            row_A,
            B
        )


        
        with lock:

            C[i] = result

            completed_rows[i] = True


        print(
            thread_name,
            "completed row",
            i + 1
        )


       
        time.sleep(0.03)


   
    with lock:
        thread_current_row[thread_id] = -1



def start_multiplication():

    print()
    
    print("        MATRIX MULTIPLICATION STARTED")
    

    print("Matrix A :", A.shape)
    print("Matrix B :", B.shape)
    print("Matrix C :", C.shape)
    print("Threads  :", NUM_THREADS)
    print("Tensor Framework : PyTorch")

    print()



    rows_per_thread = ROWS // NUM_THREADS


    with ThreadPoolExecutor(
        max_workers=NUM_THREADS
    ) as executor:

        futures = []


        for thread_id in range(NUM_THREADS):

            start_row = thread_id * rows_per_thread


            if thread_id == NUM_THREADS - 1:

                end_row = ROWS

            else:

                end_row = start_row + rows_per_thread


            future = executor.submit(
                multiply_rows,
                start_row,
                end_row,
                thread_id
            )


            futures.append(future)



        for future in futures:
            future.result()


    print()
   
    print("       ALL THREADS COMPLETED")
    


def animate():

   
    display_C = torch.full(
        (ROWS, COLS),
        float("nan"),
        dtype=torch.float32
    )



    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 6)
    )


    

    image_A = axes[0].imshow(
        A.numpy(),
        cmap="Blues",
        aspect="auto"
    )

    axes[0].set_title(
        "Matrix A"
    )

    axes[0].set_xlabel(
        "Columns"
    )

    axes[0].set_ylabel(
        "Rows"
    )


    
    red_line_A, = axes[0].plot(
        [0, COLS - 1],
        [0, 0],
        color="red",
        linewidth=3
    )


   

    image_B = axes[1].imshow(
        B.numpy(),
        cmap="Greens",
        aspect="auto"
    )

    axes[1].set_title(
        "Matrix B"
    )

    axes[1].set_xlabel(
        "Columns"
    )

    axes[1].set_ylabel(
        "Rows"
    )


    

    red_line_B, = axes[1].plot(
        [0, 0],
        [0, ROWS - 1],
        color="red",
        linewidth=3
    )



    result_image = axes[2].imshow(
        display_C.numpy(),
        cmap="Oranges",
        aspect="auto"
    )


    

    result_image.cmap.set_bad(
        color="white"
    )


    axes[2].set_title(
        "Matrix C (building live)"
    )

    axes[2].set_xlabel(
        "Columns"
    )

    axes[2].set_ylabel(
        "Rows"
    )


   
    fig.suptitle(
        "Matrix Multiplication in Action (100x100, "
        "4 Threads + PyTorch Tensors)",
        fontsize=14,
        fontweight="bold"
    )


    

    footer = fig.text(
        0.5,
        0.02,
        "Starting...",
        ha="center",
        fontsize=11
    )


    plt.tight_layout(
        rect=[0, 0.06, 1, 0.94]
    )


   
    worker = threading.Thread(
        target=start_multiplication
    )

    worker.start()


   
    next_row = 0

    current_column = 0

    last_time = time.time()


    

    def update(frame):

        nonlocal next_row
        nonlocal current_column
        nonlocal last_time


        
        with lock:

            progress = completed_rows.clone()

            current_C = C.clone()


        

        if next_row < ROWS:

            

            if progress[next_row]:

                display_C[next_row, :] = current_C[next_row, :]

                next_row += 1


        

        if next_row < ROWS:

            current_row = next_row

        else:

            current_row = ROWS - 1


        
       
        red_line_A.set_ydata(
            [current_row, current_row]
        )


       

        red_line_B.set_xdata(
            [current_column, current_column]
        )


        
        result_image.set_data(
            display_C.numpy()
        )


       

        current_column += 1


        if current_column >= COLS:

            current_column = 0


        
        displayed_rows = next_row


        if displayed_rows < ROWS:

            footer.set_text(
                "Progress: "
                + str(displayed_rows)
                + "/"
                + str(ROWS)
                + " rows done"
                + "    |    "
                + "Working on C["
                + str(current_row)
                + "]["
                + str(current_column)
                + "]"
                + "    |    "
                + "A["
                + str(current_row)
                + "] × B[:,"
                + str(current_column)
                + "]"
            )


            axes[2].set_title(
                "Matrix C (building live)\n"
                "Row "
                + str(displayed_rows + 1)
                + " of 100"
            )


        else:

            footer.set_text(
                "Progress: 100/100 rows done"
                + "    |    "
                + "MATRIX MULTIPLICATION COMPLETED"
            )


            axes[2].set_title(
                "Matrix C (Completed)\n"
                "100 / 100 rows"
            )


            fig.suptitle(
                "Matrix Multiplication Completed!\n"
                "100x100 | 4 Threads | PyTorch Tensors",
                fontsize=14,
                fontweight="bold"
            )


        return [
            result_image,
            red_line_A,
            red_line_B,
        ]


    animation = FuncAnimation(
        fig,
        update,
        interval=ANIMATION_SPEED,
        blit=False,
        cache_frame_data=False
    )


    plt.show()


 
    worker.join()




def verify_result():

    print()
    
    print("             VERIFYING RESULT")
    



    expected = torch.matmul(
        A,
        B
    )


   
    if torch.allclose(
        C,
        expected
    ):

        print()
        print("RESULT VERIFIED SUCCESSFULLY!")
        print("Threaded result is correct.")
        print("PyTorch verification: PASSED")

    else:

        print()
        print("RESULT VERIFICATION FAILED!")



if __name__ == "__main__":

    print()
    
    print("       100 x 100 MATRIX MULTIPLICATION")
  
    print()
    print("Tensor Framework : PyTorch")
    print("Matrix Size      : 100 x 100")
    print("Number of Threads: 4")
    print()


   

    animate()


   

    verify_result()


   

    print()
    print("First 5 x 5 of Matrix A:")
    print(A[:5, :5])


    print()
    print("First 5 x 5 of Matrix B:")
    print(B[:5, :5])


    print()
    print("First 5 x 5 of Result Matrix C:")
    print(C[:5, :5])


    
