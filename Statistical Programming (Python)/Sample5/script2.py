if __name__ == "__main__":
  import re
  import os
  import dask
  import time
  import pandas as pd
  import dask.bag as db
  import dask.multiprocessing

  # Set evaluation method and resources for dask
  nworkers = int(os.getenv("SLURM_NTASKS"))
  dask.config.set(scheduler = "processes", num_workers = nworkers)

  # Make function to get only lines whose
  # title has a match to regex in them
  def db_filter_title(line, regex = "Barack_Obama"):
    vals = line.split(" ")
    if len(vals) < 6:
        return(False)
    tmp = re.search(regex, vals[3])
    if tmp is None:
        return(False)
    return(True)
  
  # Make function to get only lines whose
  # language has a match to regex in them
  def db_filter_lang(line, regex = "en"):
    vals = line.split(" ")
    if len(vals) < 6:
        return(False)
    tmp = re.search(regex, vals[2])
    if tmp is None:
        return(False)
    return(True)
  
  # Make function to split the lines into a list
  def make_lst(line):
    return(list(line.split(" ")))

  # Directory for the data files
  data_dir = "/datadir/"

  # Start timer
  t0 = time.time()

  # Read in data
  x = db.read_text(data_dir + "part-0*gz", compression = "gzip")
  # Filter data
  dtypes = {"date": "object",
            "time": "object",
            "language": "object",
            "webpage": "object",
            "hits": "float64",
            "size": "float64"}
  filtered = x.filter(db_filter_title).filter(db_filter_lang)
  df = filtered.map(make_lst).to_dataframe(dtypes)
  results = df.compute()

  # End timer
  t1 = time.time()

  # Change to a datetime variable
  results["date"] = pd.to_datetime(results["date"], format='%Y%m%d')
  # Change to an hours, minutes, seconds time variable
  results["time"] = pd.to_datetime(results["time"], format='%H%M%S').dt.time

  # Write to csv
  results.to_csv("results2.csv", index = False)

  # Show total time for dask portion
  print("Time to run dask code:")
  print(pd.to_datetime(t1-t0, unit = "s").strftime('%H:%M:%S'))