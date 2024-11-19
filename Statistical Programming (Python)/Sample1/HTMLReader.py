def HTMLReader(PATH):
    '''
    Used to read in an html file and
    return a parseable BeautifulSoup object
    '''

    # Import packages, bs4 is used to help parse through the html
    try:
        from bs4 import BeautifulSoup as bs
    except ModuleNotFoundError:
        import os
        with open('requirements.txt') as f:
            lines = f.readlines()
            install_bs4_command = lines[1].strip("\n")
            os.system(install_bs4_command)

    # Open the file with latin-1 encoding
    # (I was getting errors when using UTF-8)
    html_file = open(PATH, 'r', encoding='latin-1')

    #Read the data and assign to variable
    html = html_file.read()

    # Turn raw html into BeautifulSoup object for parsing
    soup = bs(html, 'html.parser')

    # Return BeautifulSoup object
    return(soup)