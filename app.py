
import streamlit as st
import preprocessor
import helper
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pandas as pd
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

# Function to display word cloud for each topic
def display_wordcloud_for_topic(lda_model, feature_names, num_words=5):
    for idx, topic in enumerate(lda_model.components_):
        st.subheader(f"Topic #{idx+1}")
        word_freq = {feature_names[i]: topic[i] for i in topic.argsort()[:-num_words - 1:-1]}
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
        st.image(wordcloud.to_array(), caption=f'Topic #{idx+1} Word Cloud')


# Sidebar title and file uploader
st.sidebar.title("WhatsApp Chat Analyzer")
uploaded_file = st.sidebar.file_uploader("Choose a file")

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)
    df.to_csv('data.csv')

    # Fetch unique users
    user_list = df['user'].unique().tolist()
    if 'group_notification' in user_list:
        user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")

    # Show analysis wrt selected user
    selected_user = st.sidebar.selectbox("Show analysis with respect to", user_list)

    if st.sidebar.button("Show Analysis"):
        # Stats Area
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
        st.title("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Messages", num_messages)
        with col2:
            st.metric("Total Words", words)
        with col3:
            st.metric("Media Shared", num_media_messages)
        with col4:
            st.metric("Links Shared", num_links)

        # Monthly timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Daily timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Activity map
        st.title('Activity Map')
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Most Busy Day")
            busy_day = helper.week_activity_map(selected_user, df)
            st.bar_chart(busy_day)
        with col2:
            st.subheader("Most Busy Month")
            busy_month = helper.month_activity_map(selected_user, df)
            st.bar_chart(busy_month)

        # Weekly Activity Map
        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        st.write(sns.heatmap(user_heatmap))

        # Display the busiest users in the group (Group level)
        if selected_user == 'Overall':
            st.title('Most Busy Users')
            x, new_df = helper.most_busy_users(df)
            st.bar_chart(x)
            st.dataframe(new_df)

        # WordCloud
        st.title("Wordcloud")
        df_wc = helper.create_wordcloud(selected_user, df)
        st.image(df_wc.to_array())

        # Most common words
        st.title('Most Common Words')
        most_common_df = helper.most_common_words(selected_user, df)
        st.bar_chart(most_common_df.set_index(0))

        # Emoji Analysis
        st.header("Emoji Analysis")
        emoji_df = helper.emoji_count(selected_user, df)
        if not emoji_df.empty:  # Check if emoji_df is not empty
         col1, col2 = st.columns(2)
         with col1:
          st.dataframe(emoji_df)
         with col2:
          fig, ax = plt.subplots(figsize=(8, 8))
          ax.pie(emoji_df['Count'].head(), labels=emoji_df['Emoji'].head(), autopct="%0.2f")
          ax.set_title('Emoji Distribution')
          st.pyplot(fig)
        else:
          st.error(f"No Emoji shared by {selected_user}")



        st.title("WhatsApp Chat Topic Modeling")

        # Perform topic modeling
        lda_model, vectorizer = helper.perform_topic_modeling(df['message'])

        # Display word cloud for each topic
        st.subheader("Word Clouds for Each Topic")
        display_wordcloud_for_topic(lda_model, vectorizer.get_feature_names_out())

        # Display topic distribution
        st.subheader("Topic Distribution")
        topic_distribution = pd.DataFrame(lda_model.transform(vectorizer.transform(df['message'])))
        st.bar_chart(topic_distribution)

        st.title("Display List of Links")

        # Call your function to get the list of links
        links_df = helper.list_of_links(df)

        # Display the list of links
        st.dataframe(links_df)

        st.title("Top 10 Dangerous Words")

        top_dangerous_words = helper.dangerous_words_list(df)
        if top_dangerous_words:
            dangerous_words_df = pd.DataFrame(top_dangerous_words, columns=["Word", "Frequency"])
            st.write("Top Frequent Dangerous Words:")
            st.table(dangerous_words_df.head(10))
        else:
            st.info("No dangerous words found in the messages.")
