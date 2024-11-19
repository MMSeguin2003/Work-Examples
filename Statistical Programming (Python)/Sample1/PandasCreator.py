def PandasCreator(html):
    '''
    Used to create a pandas dataframe with
    rows for each publication from a scholar
    and a column for the title, authors,
    journal information, year of publication,
    and number of citations from an html
    file which can be provided as a path
    or a BeautifulSoup python object and
    return that pandas dataframe
    '''

    # Import pandas to make dataframe object at the end
    try:
      import pandas as pd
    except ModuleNotFoundError:
      import os
      with open('requirements.txt') as f:
            lines = f.readlines()
            install_pandas_command = lines[2].strip("\n")
            os.system(install_pandas_command)

    # Check to see if a filepath or an html object was
    # given and import html if needed
    if type(html) is str:
        from HTMLReader import HTMLReader as htmlreader
        html = htmlreader(html)
    
    # Find the table corresponding to the publications.
    #
    # Note: From inspecting the webpage we can see
    # Google Scholar seems to use the gsc_a_t tag for this
    #
    # Since we know there will be only one matching table
    # we can use this to find the table and extract the
    # data from the list
    table = html.body.find_all("table", attrs = {"id": "gsc_a_t"})[0]

    # Make a list of all the rows from this table
    # Note: tr is table row
    rows = table.find_all("tr")

    # Make a matrix that has the columns from the table
    # by searching for all td (which I think is table div)
    elem_matrix = [row.find_all("td") for row in rows]

    # Remove all empty objects
    # Note: the title of the table has no td, making it empty here
    elem_matrix = [elem for elem in elem_matrix if elem != []]

    # Get the publication details from each row
    # which is in the first element of the row
    # Note: this has the title, authors, and journal all combined
    publications = [row[0].contents for row in elem_matrix]

    # Get the title from each publication
    # which is in the first element of the
    # publication contents
    titles = [pub[0].contents[0] for pub in publications]

    # Get the author from each publication
    # which is in the second element of the
    # publication contents
    authors = [pub[1].contents[0] for pub in publications]

    # Get the journal information from each publication
    # which is in the third element of the
    # publication contents
    journals = [pub[2].contents[0] for pub in publications]

    # Get the number of citations from each row
    # which is in the second element of the row
    num_citations = [row[1].a.contents[0] for row in elem_matrix]

    # Get the year published from each row
    # which is in the third element of the row
    year_published = [row[2].span.contents[0] for row in elem_matrix]

    # Create pandas data frame
    pd_table = pd.DataFrame(
        {"Title": titles,
         "Authors": authors,
         "Journal Information": journals,
         "Year Published": year_published,
         "Citation Count": num_citations
         }
                             )
    
    # Return our pandas data frame
    return(pd_table)