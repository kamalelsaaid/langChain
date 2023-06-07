
# ice_breaker

A repository for learning LangChain by building a generative ai application.

This is a web applicaiton crawling Linkedin & Twitter data about a person an customize an ice breaker with them. 

[![udemy](https://img.shields.io/badge/LangChain%20Udemy%20Course-Coupon%20%2412.99-brightgreen)](https://www.udemy.com/course/langchain/?referralCode=D981B8213164A3EA91AC)

## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

`PYTHONPATH=/{YOUR_PATH_TO_PROJECT}/ice_breaker`

`OPENAI_API_KEY`

`PROXYCURL_API_KEY`

`SERPAPI_API_KEY`

`TWITTER_API_KEY`

`TWITTER_API_SECRET`

`TWITTER_ACCESS_TOKEN`

`TWITTER_ACCESS_SECRET`
## Run Locally

Clone the project

```bash
  git clone [https://github.com/emarco177/ice_breaker/blob/main/](https://github.com/emarco177/ice_breaker.git
```

Go to the project directory

```bash
  cd ice_breaker
```

Install dependencies

```bash
  pipenv install
```

Start the flask server

```bash
  pipenv run app.py
```


## Running Tests

To run tests, run the following command

```bash
  pipenv run pytest .
```


