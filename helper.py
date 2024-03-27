from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


extract = URLExtract()

def fetch_stats(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # fetch the number of messages
    num_messages = df.shape[0]

    # fetch the total number of words
    words = []
    for message in df['message']:
        words.extend(message.split())

    # fetch number of media messages
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]

    # fetch number of links shared
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages,len(words),num_media_messages,len(links)

def most_busy_users(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return x,df

def create_wordcloud(selected_user,df):

    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

def most_common_words(selected_user,df):

    f = open('stop_hinglish.txt','r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = []

    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

def emoji_count(selected_user,df):
    if selected_user!='Overall':
        df=df[df['user']==selected_user]
        
    def emoji_extract(msg):
        emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                "]+", flags=re.UNICODE)
        emoji=emoji_pattern.findall(msg) # no emoji
        # Split combined emojis into individual elements
        emoji = [char for emoji in emoji for char in emoji]
        return emoji
    chat_emojis=[]
    for msg in df['message']:
        for i in msg.split():
            w=emoji_extract(i)
            if len(w)!=0:
                chat_emojis.extend(w)
    emoji_df=pd.DataFrame(Counter(chat_emojis).most_common(20)).reindex().rename(
        columns={
            0:"Emoji",
            1:"Count"
        }
    )
    return emoji_df



def monthly_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline

def daily_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()

    return daily_timeline

def week_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()

def month_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()

def activity_heatmap(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    return user_heatmap


# Function to perform topic modeling
def perform_topic_modeling(text_data, num_topics=3):
    # Vectorize the text
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(text_data)
    
    # Perform Latent Dirichlet Allocation
    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda_model.fit(X)
    
    return lda_model, vectorizer


def list_of_links(df):
    # Fetch number of links shared
    links = []
    for message in df['message']:
        extracted_links = extract.find_urls(message)
        # Filter links that start with 'https'
        filtered_links = [link for link in extracted_links if link.startswith('https')]
        links.extend(filtered_links)
    
    links_df = pd.DataFrame(links, columns=['Links'])
    return links_df

def dangerous_words_list(df):
    # Read words from the words.txt file
    with open('words_and_synonyms.txt', 'r') as file:
        dangerous_words = set(file.read().split())

    # Tokenize messages and count occurrences of dangerous words
    word_counter = Counter()
    for message in df['message']:
        # Tokenize message by whitespace and remove non-alphanumeric characters
        words = re.findall(r'\b\w+\b', message.lower())
        # Count occurrences of dangerous words
        word_counter.update(word for word in words if word in dangerous_words)

    # Get top 10 frequent dangerous words
    top_dangerous_words = word_counter.most_common(10)
    return top_dangerous_words














