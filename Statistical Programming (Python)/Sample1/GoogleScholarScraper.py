def GoogleScholarScraper(UserID):
    '''
    Used to create the Google Scholar
    Profile link given a Scholar ID
    then download the html page and
    read it into python, returning
    it as a BeautifulSoup object
    '''

    # Import subprocess package so we can make
    # query and import HTMLReader to read in
    # the downloaded html
    import subprocess
    from HTMLReader import HTMLReader as htmlreader

    # Deconstruct sections of link based on
    # looking at some example links
    base_link = "https://scholar.google.com"
    query_citations = "/citations?user="
    ending_filters = "&amp;hl=en&amp;oi=ao"

    # Construct link based on inputted UserID
    Scholar_Profile_Link = (base_link +
                            query_citations +
                            UserID +
                            ending_filters)

    # Download html
    htmlpath = "scholar_profile.html"
    subprocess.run(["curl", "-o " + htmlpath,
                    Scholar_Profile_Link],
                    capture_output=True)
    
    # Import to python using our previous function
    soup = htmlreader(htmlpath)

    # Return BeautifulSoup object
    return(soup)
