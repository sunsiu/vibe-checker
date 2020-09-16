import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import json
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

KEY_PATH = 'C:/Users/Sunny/Dropbox/College/DS 2500/Final Project/spotify_keys.json'
with open(KEY_PATH, 'r') as f:
  creds = json.load(f)

manager = SpotifyClientCredentials(client_id=creds['client_id'],client_secret=creds['client_secret'])
sp = spotipy.Spotify(client_credentials_manager=manager)


def get_playlist_tracks(p_id):
  """
  Get the individual tracks given a playlist id
  """
  print('Fetching tracks...')
  try:
    playlist = sp.playlist_tracks(p_id)
  except:
    print('Could not find playlist')
    return None

  songs = playlist['items']
  while playlist['next']:
    playlist = sp.next(playlist)
    songs.extend(playlist['items'])

  playlist_tracks = []
  for song in songs:
    track = song['track']
    track_info = {
        'id': track['id'] if track['id'] else 'None',
        'name': track['name'],
        'artist': track['artists'][0]['name']
    }
    playlist_tracks.append(track_info)

  return pd.DataFrame(playlist_tracks)


def get_tracks(user, p_name):
  """
  Gets tracks based on username and playlist name. The user must own
  own the playlist for this to work.
  """
  playlist_id = None
  my_playlists = sp.user_playlists(user)
  for playlist in my_playlists['items']:
    if playlist['name'].lower() == p_name:
      playlist_id = playlist['id']
      break

  if not playlist_id:
    sys.exit('Playlist not found')
    return

  return get_playlist_tracks(playlist_id)


def add_valence(tracks_df, features):
  """
  Adds valence features to the dataframe 1-to-1
  """
  for i, track in tracks_df.iterrows():
    if track['id'] == 'None':
      tracks_df.loc[i, 'valence'] = float(0)
    else:
      tracks_df.loc[i, 'valence'] = features[i]['valence']


def get_valence(tracks_df):
  """
  Takes the given df and adds a normalized(-1, 1) valence as a column
  """
  row_size = tracks_df.shape[0]
  splits = (row_size / 50) + 1
  track_ids = np.array_split(tracks_df['id'].to_numpy(), splits)

  features = []
  for id_list in track_ids:
    features += sp.audio_features(id_list)

  add_valence(tracks_df, features)

  # Normalize valence scores to match our Sentiment Analysis scores (-1, 1)
  tracks_df['valence'] = tracks_df['valence'].apply(lambda x: (x * 2) - 1 if x != 0 else 0)
  return tracks_df


def deviation_chart(df, title, col='valence'):
  cleaned = df.dropna()
  cleaned['sign'] = cleaned[col].apply(lambda x: 'r' if np.sign(x) < 0 else 'lightblue')
  cleaned = cleaned.sort_values(by=[col], ascending=True)

  height = .15 * cleaned.shape[0]
  plt.figure(figsize=(6, height), dpi=80)
  plt.margins(y=.01)
  plt.hlines(y=cleaned.name, xmin=0, xmax=cleaned[col], color=cleaned.sign, linewidth=6)
  plt.axvline(color='black', alpha=.3)  # Plot a vertical line at 0
  plt.title(title)
  plt.ylabel("Song Name")
  plt.xlabel(col)
  plt.show()

def pie_chart(df, title, col='valence'):
  labels = ['Happy', 'Sad']

  cleaned = df.dropna()
  size = cleaned.shape[0]
  cleaned['sign'] = cleaned[col].apply(lambda x: 'r' if np.sign(x) < 0 else 'lightblue')
  pos = cleaned['sign'].value_counts()['lightblue']
  neg = cleaned['sign'].value_counts()['r']
  percentages = [pos, neg]

  fig, ax = plt.subplots()
  ax.pie(percentages, labels=labels,
         colors=['lightblue', 'r'], autopct='%1.0f%%')
  ax.legend()
  ax.set_title(title)
  plt.show()


def avg_with_error(score, col):
  """
  Get the average score for a given column.
  Ignores rows that have no value
  """
  size = col.shape[0]
  errors = col.isna().sum()
  result = score / (size - errors)
  return result


def report(playlist, scored):
  """
  Reports the total and average scores.
  """
  print(playlist + " Report")
  print("----------------------------")
  v_total = scored.valence.sum()
  avg_valence = avg_with_error(v_total, scored.valence)
  print('Valence Total: %.3f' % v_total)
  print('Valence Mean: %.3f' % avg_valence)
  print('Verdict: ', 'Happy' if np.sign(v_total) > 0 else 'Sad')
  print('\n')


# Prompt user for username and playlist name
while True:
  choice = input('Do you know the playlist id? (Y/N): ')
  if choice.lower() == 'y':
    p_id = input('Enter id: ')
    playlist_name = ''
    tracks = get_playlist_tracks(p_id)
    break
  elif choice.lower() == 'n':
    username = input('Spotify Username: ')
    playlist_name = input('Playlist Name: ').lower()
    tracks = get_tracks(username, playlist_name)
    break
print('Scoring playlist')
tracks_scored = get_valence(tracks)

report('', tracks_scored)
deviation_chart(tracks_scored, 'Song Valences')
pie_chart(tracks_scored, playlist_name + ' Makeup')