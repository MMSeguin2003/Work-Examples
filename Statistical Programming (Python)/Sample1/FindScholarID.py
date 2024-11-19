def FindScholarID(html, returnelement = False):
    '''
    Used to find the Scholar ID or optionally
    the whole element containing it in an
    html file which can be provided as a path
    or a BeautifulSoup python object and
    return either the element with the
    Scholar ID or just the Scholar ID
    '''

    # Check to see if a filepath or an html object was
    # given and import html if needed
    if type(html) is str:
        from HTMLReader import HTMLReader as htmlreader
        html = htmlreader(html)
    
    # Find all of the tables in the html
    tables = html.body.find_all("table")

    # Find all of the hyperlinks inside
    # each table (a is an html hyperlink tag)
    links = [table.a for table in tables]

    # Get the elements from links as strings
    elements = [str(link) for link in links]

    # Search for which element has "user="
    element = elements["user=" in elements]

    # If we want to return the whole element return
    if returnelement:
        return(element)

    # Otherwise we want just the ID so find
    # where the query field starts for it
    start_index = element.find("user=")

    # Define a function that finds a given
    # query field given a string and an
    # index to start at
    def LocateField(string, start):
        '''
        Used to find a query field from a link
        given a starting point from the link
        and the link as a string
        '''

        # Create start and end point variables
        begin = ""
        end = ""

        # Initialize iterators
        i, j = start, start

        # While we haven't found both
        # values keep iterating
        while begin == "" or end == "":
            # If the current value is the signal
            # of the assignment of a query field
            # and we haven't already assigned a
            # begin point then assign begin
            if string[i] == "=" and begin == "":
                begin = i + 1
            # Otherwise move one character forward
            # Note: we are moving forward because
            # Using find on a string gives the
            # location of the first entry of what
            # We are looking for so the "=" is later
            else:
                i += 1
            # If the current value is the signal
            # of the start of another query
            # field and we haven't already
            # assigned an end point then assign end
            if string[j] == "&" and end == "":
                end = j
            # Otherwise move one character forward
            else:
                j += 1
        # Return the string from the starting
        # to the end point found
        return(string[begin:end])
    
    # Use our locate function to find the user
    # query field which is the Scholar ID
    UserID = LocateField(element, start_index)

    # Return the Scholar ID
    return(UserID)