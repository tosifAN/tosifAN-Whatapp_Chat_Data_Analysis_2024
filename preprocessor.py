import pandas as pd
import re

def preprocess(string, key='12hr', custom_time_format=''):
    '''Converts raw string into a Data Frame'''
    
    split_formats = {
        '12hr' : '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s[APap][mM]\s-\s',
        '24hr' : '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s',
        'custom' : custom_time_format
    }
    
    raw_string = string
    user_msg = re.split(split_formats[key], raw_string)[1:]
    date_time = re.findall(split_formats[key], raw_string)
    
    # Preprocess datetime strings to remove non-numeric characters
    date_time = [re.sub(r'[^\d\s:/]', '', dt) for dt in date_time]
        
    df = pd.DataFrame({'date': date_time, 'user_msg': user_msg})
    
    # Define possible datetime formats
    possible_formats = [
        '%d/%m/%y %I:%M %p ',
        '%d/%m/%Y %I:%M %p ',
        '%d/%m/%y %H:%M ',
        '%d/%m/%Y %H:%M '
    ]
    
    for format in possible_formats:
        try:
            df['date'] = pd.to_datetime(df['date'], format=format)
            break  # If successful, exit the loop
        except ValueError:
            pass  # If unsuccessful, try the next format
    
    # Ensure 'date_time' column is converted to datetime format
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Drop rows with NaT (Not a Time) values
    df = df.dropna(subset=['date'])
    
    usernames = []
    msgs = []
    for i in df['user_msg']:
        a = re.split('([\w\W]+?):\s', i)
        if(a[1:]):
            usernames.append(a[1])
            msgs.append(a[2])
        else:
            usernames.append("group_notification")
            msgs.append(a[0])

    df['user'] = usernames
    df['message'] = msgs

    df.drop('user_msg', axis=1, inplace=True)
    '''
    # Add additional date-related columns
    df['day'] = df['date_time'].dt.strftime('%a')
    df['month'] = df['date_time'].dt.strftime('%b')
    df['year'] = df['date_time'].dt.year
    df['date'] = df['date_time'].apply(lambda x: x.date())
    '''
    #df['date'] = df['date_time'].apply(lambda x: x.date())

    # updating 
    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    period = []
    for hour in df[['day_name', 'hour']]['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))

    df['period'] = period

    return df
    
