if __name__ == "__main__":
  import re
  import os
  import dask
  import pandas as pd

  # Set evaluation method and resources for dask
  nworkers = int(os.getenv("SLURM_NTASKS"))
  dask.config.set(scheduler = "processes", num_workers = nworkers)

  # Make function to read the files and get only
  # lines that have "Barack_Obama" in them
  def read_and_filter(filename):
    x = pd.read_csv(filename,
                    sep = "\\s+",
                    dtype = {0: "str",
                             1: "str"
                            },
                    compression = "gzip",
                    header = None,
                    quoting = 3)
    x = x[[re.search("Barack_Obama", str(i)) is not None for i in x[3]]]
    return(x)

  # Directory for the data files
  data_dir = "/tmp/subdir"

  # Make a list of the paths to the files
  file_names = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".gz")]

  # Create tasks list
  tasks = [dask.delayed(read_and_filter)(file_name) for file_name in file_names]

  # Compute results
  results = dask.compute(tasks)[0]

  # Combine all resulting data frames
  combined_results = pd.concat(results, ignore_index = True)
  # Change to a datetime variable
  combined_results[0] = pd.to_datetime(combined_results[0], format='%Y%m%d')
  # Change to an hours, minutes, seconds time variable
  combined_results[1] = pd.to_datetime(combined_results[1], format='%H%M%S').dt.time

  # Write to csv
  combined_results.to_csv("results1.csv", index = False)
  