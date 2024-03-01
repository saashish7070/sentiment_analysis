import streamlit as st
import pickle
from utils import process_tweet
def naive_bayes_predict(tweet, logprior, loglikelihood):
    '''
    Input:
        tweet: a string
        logprior: a number
        loglikelihood: a dictionary of words mapping to numbers
    Output:
        p: the sum of all the logliklihoods of each word in the tweet (if found in the dictionary) + logprior (a number)

    '''
    # process the tweet to get a list of words
    word_l = process_tweet(tweet)

    # initialize probability to zero
    p = 0

    # add the logprior
    p += logprior

    for word in word_l:

        # check if the word exists in the loglikelihood dictionary
        if word in loglikelihood:
            # add the log likelihood of that word to the probability
            p += loglikelihood[word]

    return p


with open('model_params.pkl', 'rb') as f:
    logprior, loglikelihood = pickle.load(f)

def main():
    st.title('Sentiment Analysis Using Naive Bayes')

    tweet = st.text_input('Enter your tweet:')

    # Button to trigger sentiment analysis
    if st.button('Analyze Sentiment'):
        sentiment = naive_bayes_predict(tweet, logprior, loglikelihood)
        print(sentiment)
        # st.write(f'Tweet sentiment: {sentiment}')
        if sentiment>0 :
            st.write(f'Tweet sentiment: Positive Tweet with the loglikelihood of {sentiment} ')
        else:
            st.write(f'Tweet sentiment: Negative Tweet with the loglikelihood of {sentiment}')


if __name__ == '__main__':
    main()
