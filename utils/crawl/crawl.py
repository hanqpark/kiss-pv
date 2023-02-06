import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from multiprocessing import Process


def crawling():
    driver = webdriver.Chrome("chromedriver")
    driver.get("https://www.kbsec.com/go.able?linkcd=m04010008")
    res = driver.find_element(By.XPATH, "# /html/body/div[2]/div[4]/div[3]/table/tbody/tr[1]/td[1]/text()[1]")
    print(res)
    
if __name__=='__main__':
    p1 = Process(target=crawling) #함수 1을 위한 프로세스
    p1.start()