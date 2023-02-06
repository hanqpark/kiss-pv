from selenium import webdriver
from bs4 import BeautifulSoup

# setup the webdriver
driver = webdriver.Chrome('chromedriver_mac_arm64.zip')

# navigate to the website
driver.get("https://www.kbsec.com/go.able?linkcd=m04010008")

# wait for the page to load
driver.implicitly_wait(10)

# parse the html content
soup = BeautifulSoup(driver.page_source, "html.parser")

# find all headings
print(soup)
headings = soup.select("#pList1")

# close the webdriver
driver.quit()
