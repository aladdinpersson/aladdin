import torch
import timeit


def find_optimal_workers(
    loader,
    start_workers=0,
    max_workers=12,
    worker_step=1,
    iterations_per_worker=20,
    num_averages=3,
    set_pin_memory=True,
    verbose=True,
):
    """
        Find the (approximately) best number of workers to use for a particular DataLoader.

        Parameters
        ----------
        loader: DataLoader Class
            DataLoader for the dataset you want to check on
        start_workers: Integer
            How many workers to start from
        max_workers: Integer
            How many workers to end at
        worker_step: Integer
            Step size of number of workers
        iterations_per_worker: Integer
            The number of iterations we run each loader for, the larger the better
            in order to get a better approximation, but the longer the time.
        num_averages: Integer
            How many times we do the same experiment for specific number of workers
        set_pin_memory: Boolean
            Change DataLoader pin memory to True by default
        verbose: Boolean
            Prints information of iterations/second for every worker

        Returns
        -------
        int
            Number of workers found to be best
        """

    best_num_workers, best_iterations = None, float("-inf")

    if verbose:
        if not loader.pin_memory:
            warnings.warn(
                "Pin memory is not True, I'm setting it to True (send set_pin_memory=False is not desired)"
            )

    loader.pin_memory = set_pin_memory

    for num_workers in range(start_workers, max_workers, worker_step):
        loader.num_workers = num_workers
        average_iterations = 0

        for _ in range(num_averages):
            iterations = 0
            for i, data in enumerate(loader):
                if i == 1:
                    start_time = timeit.default_timer()

                if i > 1:
                    iterations += 1

                if i == iterations_per_worker:
                    end_time = timeit.default_timer()
                    iterations_divided_by_time = iterations / (end_time - start_time)
                    average_iterations += iterations_divided_by_time
                    break

        if verbose:
            print(
                f"Iterations/second when num_workers={num_workers} is: {average_iterations/num_averages}"
            )

        if average_iterations / num_averages > best_iterations:
            best_num_workers = num_workers
            best_iterations = average_iterations / num_averages

    return best_num_workers