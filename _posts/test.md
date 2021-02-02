# Investigating the Chess Meta: A Data Driven Approach

You put in your chess username, it looks you up in the lichess API and then gives you ratings according to aggresive vs solid, tells you your best performing openings and others to try


## Import Data and Libraries


```
cd '/content/drive/MyDrive/Chess'
```

    /content/drive/MyDrive/Chess
    


```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
%matplotlib inline

```


```
df = pd.read_csv("games.csv")
```

## A Quick Look & Transformations




```
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>rated</th>
      <th>created_at</th>
      <th>last_move_at</th>
      <th>turns</th>
      <th>victory_status</th>
      <th>winner</th>
      <th>increment_code</th>
      <th>white_id</th>
      <th>white_rating</th>
      <th>black_id</th>
      <th>black_rating</th>
      <th>moves</th>
      <th>opening_eco</th>
      <th>opening_name</th>
      <th>opening_ply</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TZJHLljE</td>
      <td>False</td>
      <td>1.504210e+12</td>
      <td>1.504210e+12</td>
      <td>13</td>
      <td>outoftime</td>
      <td>white</td>
      <td>15+2</td>
      <td>bourgris</td>
      <td>1500</td>
      <td>a-00</td>
      <td>1191</td>
      <td>d4 d5 c4 c6 cxd5 e6 dxe6 fxe6 Nf3 Bb4+ Nc3 Ba5...</td>
      <td>D10</td>
      <td>Slav Defense: Exchange Variation</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>l1NXvwaE</td>
      <td>True</td>
      <td>1.504130e+12</td>
      <td>1.504130e+12</td>
      <td>16</td>
      <td>resign</td>
      <td>black</td>
      <td>5+10</td>
      <td>a-00</td>
      <td>1322</td>
      <td>skinnerua</td>
      <td>1261</td>
      <td>d4 Nc6 e4 e5 f4 f6 dxe5 fxe5 fxe5 Nxe5 Qd4 Nc6...</td>
      <td>B00</td>
      <td>Nimzowitsch Defense: Kennedy Variation</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>mIICvQHh</td>
      <td>True</td>
      <td>1.504130e+12</td>
      <td>1.504130e+12</td>
      <td>61</td>
      <td>mate</td>
      <td>white</td>
      <td>5+10</td>
      <td>ischia</td>
      <td>1496</td>
      <td>a-00</td>
      <td>1500</td>
      <td>e4 e5 d3 d6 Be3 c6 Be2 b5 Nd2 a5 a4 c5 axb5 Nc...</td>
      <td>C20</td>
      <td>King's Pawn Game: Leonardis Variation</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>kWKvrqYL</td>
      <td>True</td>
      <td>1.504110e+12</td>
      <td>1.504110e+12</td>
      <td>61</td>
      <td>mate</td>
      <td>white</td>
      <td>20+0</td>
      <td>daniamurashov</td>
      <td>1439</td>
      <td>adivanov2009</td>
      <td>1454</td>
      <td>d4 d5 Nf3 Bf5 Nc3 Nf6 Bf4 Ng4 e3 Nc6 Be2 Qd7 O...</td>
      <td>D02</td>
      <td>Queen's Pawn Game: Zukertort Variation</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9tXo1AUZ</td>
      <td>True</td>
      <td>1.504030e+12</td>
      <td>1.504030e+12</td>
      <td>95</td>
      <td>mate</td>
      <td>white</td>
      <td>30+3</td>
      <td>nik221107</td>
      <td>1523</td>
      <td>adivanov2009</td>
      <td>1469</td>
      <td>e4 e5 Nf3 d6 d4 Nc6 d5 Nb4 a3 Na6 Nc3 Be7 b4 N...</td>
      <td>C41</td>
      <td>Philidor Defense</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



Our first consideration


```
df = (
    df.assign(
        opening_archetype=df.opening_name.map(
            lambda n: n.split(":")[0].split("|")[0].split("#")[0].strip()
        ),
        opening_moves=df.apply(lambda srs: srs['moves'].split(" ")[:srs['opening_ply']],
                                  axis=1)
    )
)
```


```
# find distribution of chess openings by colour
# are certain openings more popular at higher ratings?
# are certain openings more popular at certain time increments?
# chances of beating a higher player
# What opening should you play when playing vs a higher player? - ie, what have the best win rate at 
# Is there a similarity between palyers
```


```
len(df['opening_archetype'].unique())
```




    143




```
df['opening_archetype'].value_counts()
```




    Sicilian Defense       2632
    French Defense         1412
    Queen's Pawn Game      1233
    Italian Game            981
    King's Pawn Game        917
                           ... 
    Valencia Opening          1
    Australian Defense        1
    Doery Defense             1
    Pterodactyl Defense       1
    Global Opening            1
    Name: opening_archetype, Length: 143, dtype: int64




```

```

## How to cause an upset?


```
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>rated</th>
      <th>created_at</th>
      <th>last_move_at</th>
      <th>turns</th>
      <th>victory_status</th>
      <th>winner</th>
      <th>increment_code</th>
      <th>white_id</th>
      <th>white_rating</th>
      <th>black_id</th>
      <th>black_rating</th>
      <th>moves</th>
      <th>opening_eco</th>
      <th>opening_name</th>
      <th>opening_ply</th>
      <th>opening_archetype</th>
      <th>opening_moves</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TZJHLljE</td>
      <td>False</td>
      <td>1.504210e+12</td>
      <td>1.504210e+12</td>
      <td>13</td>
      <td>outoftime</td>
      <td>white</td>
      <td>15+2</td>
      <td>bourgris</td>
      <td>1500</td>
      <td>a-00</td>
      <td>1191</td>
      <td>d4 d5 c4 c6 cxd5 e6 dxe6 fxe6 Nf3 Bb4+ Nc3 Ba5...</td>
      <td>D10</td>
      <td>Slav Defense: Exchange Variation</td>
      <td>5</td>
      <td>Slav Defense</td>
      <td>[d4, d5, c4, c6, cxd5]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>l1NXvwaE</td>
      <td>True</td>
      <td>1.504130e+12</td>
      <td>1.504130e+12</td>
      <td>16</td>
      <td>resign</td>
      <td>black</td>
      <td>5+10</td>
      <td>a-00</td>
      <td>1322</td>
      <td>skinnerua</td>
      <td>1261</td>
      <td>d4 Nc6 e4 e5 f4 f6 dxe5 fxe5 fxe5 Nxe5 Qd4 Nc6...</td>
      <td>B00</td>
      <td>Nimzowitsch Defense: Kennedy Variation</td>
      <td>4</td>
      <td>Nimzowitsch Defense</td>
      <td>[d4, Nc6, e4, e5]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>mIICvQHh</td>
      <td>True</td>
      <td>1.504130e+12</td>
      <td>1.504130e+12</td>
      <td>61</td>
      <td>mate</td>
      <td>white</td>
      <td>5+10</td>
      <td>ischia</td>
      <td>1496</td>
      <td>a-00</td>
      <td>1500</td>
      <td>e4 e5 d3 d6 Be3 c6 Be2 b5 Nd2 a5 a4 c5 axb5 Nc...</td>
      <td>C20</td>
      <td>King's Pawn Game: Leonardis Variation</td>
      <td>3</td>
      <td>King's Pawn Game</td>
      <td>[e4, e5, d3]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>kWKvrqYL</td>
      <td>True</td>
      <td>1.504110e+12</td>
      <td>1.504110e+12</td>
      <td>61</td>
      <td>mate</td>
      <td>white</td>
      <td>20+0</td>
      <td>daniamurashov</td>
      <td>1439</td>
      <td>adivanov2009</td>
      <td>1454</td>
      <td>d4 d5 Nf3 Bf5 Nc3 Nf6 Bf4 Ng4 e3 Nc6 Be2 Qd7 O...</td>
      <td>D02</td>
      <td>Queen's Pawn Game: Zukertort Variation</td>
      <td>3</td>
      <td>Queen's Pawn Game</td>
      <td>[d4, d5, Nf3]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9tXo1AUZ</td>
      <td>True</td>
      <td>1.504030e+12</td>
      <td>1.504030e+12</td>
      <td>95</td>
      <td>mate</td>
      <td>white</td>
      <td>30+3</td>
      <td>nik221107</td>
      <td>1523</td>
      <td>adivanov2009</td>
      <td>1469</td>
      <td>e4 e5 Nf3 d6 d4 Nc6 d5 Nb4 a3 Na6 Nc3 Be7 b4 N...</td>
      <td>C41</td>
      <td>Philidor Defense</td>
      <td>5</td>
      <td>Philidor Defense</td>
      <td>[e4, e5, Nf3, d6, d4]</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>20053</th>
      <td>EfqH7VVH</td>
      <td>True</td>
      <td>1.499791e+12</td>
      <td>1.499791e+12</td>
      <td>24</td>
      <td>resign</td>
      <td>white</td>
      <td>10+10</td>
      <td>belcolt</td>
      <td>1691</td>
      <td>jamboger</td>
      <td>1220</td>
      <td>d4 f5 e3 e6 Nf3 Nf6 Nc3 b6 Be2 Bb7 O-O Be7 Ne5...</td>
      <td>A80</td>
      <td>Dutch Defense</td>
      <td>2</td>
      <td>Dutch Defense</td>
      <td>[d4, f5]</td>
    </tr>
    <tr>
      <th>20054</th>
      <td>WSJDhbPl</td>
      <td>True</td>
      <td>1.499698e+12</td>
      <td>1.499699e+12</td>
      <td>82</td>
      <td>mate</td>
      <td>black</td>
      <td>10+0</td>
      <td>jamboger</td>
      <td>1233</td>
      <td>farrukhasomiddinov</td>
      <td>1196</td>
      <td>d4 d6 Bf4 e5 Bg3 Nf6 e3 exd4 exd4 d5 c3 Bd6 Bd...</td>
      <td>A41</td>
      <td>Queen's Pawn</td>
      <td>2</td>
      <td>Queen's Pawn</td>
      <td>[d4, d6]</td>
    </tr>
    <tr>
      <th>20055</th>
      <td>yrAas0Kj</td>
      <td>True</td>
      <td>1.499698e+12</td>
      <td>1.499698e+12</td>
      <td>35</td>
      <td>mate</td>
      <td>white</td>
      <td>10+0</td>
      <td>jamboger</td>
      <td>1219</td>
      <td>schaaksmurf3</td>
      <td>1286</td>
      <td>d4 d5 Bf4 Nc6 e3 Nf6 c3 e6 Nf3 Be7 Bd3 O-O Nbd...</td>
      <td>D00</td>
      <td>Queen's Pawn Game: Mason Attack</td>
      <td>3</td>
      <td>Queen's Pawn Game</td>
      <td>[d4, d5, Bf4]</td>
    </tr>
    <tr>
      <th>20056</th>
      <td>b0v4tRyF</td>
      <td>True</td>
      <td>1.499696e+12</td>
      <td>1.499697e+12</td>
      <td>109</td>
      <td>resign</td>
      <td>white</td>
      <td>10+0</td>
      <td>marcodisogno</td>
      <td>1360</td>
      <td>jamboger</td>
      <td>1227</td>
      <td>e4 d6 d4 Nf6 e5 dxe5 dxe5 Qxd1+ Kxd1 Nd5 c4 Nb...</td>
      <td>B07</td>
      <td>Pirc Defense</td>
      <td>4</td>
      <td>Pirc Defense</td>
      <td>[e4, d6, d4, Nf6]</td>
    </tr>
    <tr>
      <th>20057</th>
      <td>N8G2JHGG</td>
      <td>True</td>
      <td>1.499643e+12</td>
      <td>1.499644e+12</td>
      <td>78</td>
      <td>mate</td>
      <td>black</td>
      <td>10+0</td>
      <td>jamboger</td>
      <td>1235</td>
      <td>ffbob</td>
      <td>1339</td>
      <td>d4 d5 Bf4 Na6 e3 e6 c3 Nf6 Nf3 Bd7 Nbd2 b5 Bd3...</td>
      <td>D00</td>
      <td>Queen's Pawn Game: Mason Attack</td>
      <td>3</td>
      <td>Queen's Pawn Game</td>
      <td>[d4, d5, Bf4]</td>
    </tr>
  </tbody>
</table>
<p>20058 rows × 18 columns</p>
</div>




```
df['avg_rating'] = (df['black_rating'] + df['white_rating'])/2
```


```
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>rated</th>
      <th>created_at</th>
      <th>last_move_at</th>
      <th>turns</th>
      <th>victory_status</th>
      <th>winner</th>
      <th>increment_code</th>
      <th>white_id</th>
      <th>white_rating</th>
      <th>black_id</th>
      <th>black_rating</th>
      <th>moves</th>
      <th>opening_eco</th>
      <th>opening_name</th>
      <th>opening_ply</th>
      <th>opening_archetype</th>
      <th>opening_moves</th>
      <th>avg_rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TZJHLljE</td>
      <td>False</td>
      <td>1.504210e+12</td>
      <td>1.504210e+12</td>
      <td>13</td>
      <td>outoftime</td>
      <td>white</td>
      <td>15+2</td>
      <td>bourgris</td>
      <td>1500</td>
      <td>a-00</td>
      <td>1191</td>
      <td>d4 d5 c4 c6 cxd5 e6 dxe6 fxe6 Nf3 Bb4+ Nc3 Ba5...</td>
      <td>D10</td>
      <td>Slav Defense: Exchange Variation</td>
      <td>5</td>
      <td>Slav Defense</td>
      <td>[d4, d5, c4, c6, cxd5]</td>
      <td>1345.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>l1NXvwaE</td>
      <td>True</td>
      <td>1.504130e+12</td>
      <td>1.504130e+12</td>
      <td>16</td>
      <td>resign</td>
      <td>black</td>
      <td>5+10</td>
      <td>a-00</td>
      <td>1322</td>
      <td>skinnerua</td>
      <td>1261</td>
      <td>d4 Nc6 e4 e5 f4 f6 dxe5 fxe5 fxe5 Nxe5 Qd4 Nc6...</td>
      <td>B00</td>
      <td>Nimzowitsch Defense: Kennedy Variation</td>
      <td>4</td>
      <td>Nimzowitsch Defense</td>
      <td>[d4, Nc6, e4, e5]</td>
      <td>1291.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>mIICvQHh</td>
      <td>True</td>
      <td>1.504130e+12</td>
      <td>1.504130e+12</td>
      <td>61</td>
      <td>mate</td>
      <td>white</td>
      <td>5+10</td>
      <td>ischia</td>
      <td>1496</td>
      <td>a-00</td>
      <td>1500</td>
      <td>e4 e5 d3 d6 Be3 c6 Be2 b5 Nd2 a5 a4 c5 axb5 Nc...</td>
      <td>C20</td>
      <td>King's Pawn Game: Leonardis Variation</td>
      <td>3</td>
      <td>King's Pawn Game</td>
      <td>[e4, e5, d3]</td>
      <td>1498.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>kWKvrqYL</td>
      <td>True</td>
      <td>1.504110e+12</td>
      <td>1.504110e+12</td>
      <td>61</td>
      <td>mate</td>
      <td>white</td>
      <td>20+0</td>
      <td>daniamurashov</td>
      <td>1439</td>
      <td>adivanov2009</td>
      <td>1454</td>
      <td>d4 d5 Nf3 Bf5 Nc3 Nf6 Bf4 Ng4 e3 Nc6 Be2 Qd7 O...</td>
      <td>D02</td>
      <td>Queen's Pawn Game: Zukertort Variation</td>
      <td>3</td>
      <td>Queen's Pawn Game</td>
      <td>[d4, d5, Nf3]</td>
      <td>1446.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9tXo1AUZ</td>
      <td>True</td>
      <td>1.504030e+12</td>
      <td>1.504030e+12</td>
      <td>95</td>
      <td>mate</td>
      <td>white</td>
      <td>30+3</td>
      <td>nik221107</td>
      <td>1523</td>
      <td>adivanov2009</td>
      <td>1469</td>
      <td>e4 e5 Nf3 d6 d4 Nc6 d5 Nb4 a3 Na6 Nc3 Be7 b4 N...</td>
      <td>C41</td>
      <td>Philidor Defense</td>
      <td>5</td>
      <td>Philidor Defense</td>
      <td>[e4, e5, Nf3, d6, d4]</td>
      <td>1496.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>20053</th>
      <td>EfqH7VVH</td>
      <td>True</td>
      <td>1.499791e+12</td>
      <td>1.499791e+12</td>
      <td>24</td>
      <td>resign</td>
      <td>white</td>
      <td>10+10</td>
      <td>belcolt</td>
      <td>1691</td>
      <td>jamboger</td>
      <td>1220</td>
      <td>d4 f5 e3 e6 Nf3 Nf6 Nc3 b6 Be2 Bb7 O-O Be7 Ne5...</td>
      <td>A80</td>
      <td>Dutch Defense</td>
      <td>2</td>
      <td>Dutch Defense</td>
      <td>[d4, f5]</td>
      <td>1455.5</td>
    </tr>
    <tr>
      <th>20054</th>
      <td>WSJDhbPl</td>
      <td>True</td>
      <td>1.499698e+12</td>
      <td>1.499699e+12</td>
      <td>82</td>
      <td>mate</td>
      <td>black</td>
      <td>10+0</td>
      <td>jamboger</td>
      <td>1233</td>
      <td>farrukhasomiddinov</td>
      <td>1196</td>
      <td>d4 d6 Bf4 e5 Bg3 Nf6 e3 exd4 exd4 d5 c3 Bd6 Bd...</td>
      <td>A41</td>
      <td>Queen's Pawn</td>
      <td>2</td>
      <td>Queen's Pawn</td>
      <td>[d4, d6]</td>
      <td>1214.5</td>
    </tr>
    <tr>
      <th>20055</th>
      <td>yrAas0Kj</td>
      <td>True</td>
      <td>1.499698e+12</td>
      <td>1.499698e+12</td>
      <td>35</td>
      <td>mate</td>
      <td>white</td>
      <td>10+0</td>
      <td>jamboger</td>
      <td>1219</td>
      <td>schaaksmurf3</td>
      <td>1286</td>
      <td>d4 d5 Bf4 Nc6 e3 Nf6 c3 e6 Nf3 Be7 Bd3 O-O Nbd...</td>
      <td>D00</td>
      <td>Queen's Pawn Game: Mason Attack</td>
      <td>3</td>
      <td>Queen's Pawn Game</td>
      <td>[d4, d5, Bf4]</td>
      <td>1252.5</td>
    </tr>
    <tr>
      <th>20056</th>
      <td>b0v4tRyF</td>
      <td>True</td>
      <td>1.499696e+12</td>
      <td>1.499697e+12</td>
      <td>109</td>
      <td>resign</td>
      <td>white</td>
      <td>10+0</td>
      <td>marcodisogno</td>
      <td>1360</td>
      <td>jamboger</td>
      <td>1227</td>
      <td>e4 d6 d4 Nf6 e5 dxe5 dxe5 Qxd1+ Kxd1 Nd5 c4 Nb...</td>
      <td>B07</td>
      <td>Pirc Defense</td>
      <td>4</td>
      <td>Pirc Defense</td>
      <td>[e4, d6, d4, Nf6]</td>
      <td>1293.5</td>
    </tr>
    <tr>
      <th>20057</th>
      <td>N8G2JHGG</td>
      <td>True</td>
      <td>1.499643e+12</td>
      <td>1.499644e+12</td>
      <td>78</td>
      <td>mate</td>
      <td>black</td>
      <td>10+0</td>
      <td>jamboger</td>
      <td>1235</td>
      <td>ffbob</td>
      <td>1339</td>
      <td>d4 d5 Bf4 Na6 e3 e6 c3 Nf6 Nf3 Bd7 Nbd2 b5 Bd3...</td>
      <td>D00</td>
      <td>Queen's Pawn Game: Mason Attack</td>
      <td>3</td>
      <td>Queen's Pawn Game</td>
      <td>[d4, d5, Bf4]</td>
      <td>1287.0</td>
    </tr>
  </tbody>
</table>
<p>20058 rows × 19 columns</p>
</div>




```
df['rating_group'] = pd.cut(df['avg_rating'], 10, labels=False)
```


```
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>rated</th>
      <th>created_at</th>
      <th>last_move_at</th>
      <th>turns</th>
      <th>victory_status</th>
      <th>winner</th>
      <th>increment_code</th>
      <th>white_id</th>
      <th>white_rating</th>
      <th>black_id</th>
      <th>black_rating</th>
      <th>moves</th>
      <th>opening_eco</th>
      <th>opening_name</th>
      <th>opening_ply</th>
      <th>opening_archetype</th>
      <th>opening_moves</th>
      <th>avg_rating</th>
      <th>rating_group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TZJHLljE</td>
      <td>False</td>
      <td>1.504210e+12</td>
      <td>1.504210e+12</td>
      <td>13</td>
      <td>outoftime</td>
      <td>white</td>
      <td>15+2</td>
      <td>bourgris</td>
      <td>1500</td>
      <td>a-00</td>
      <td>1191</td>
      <td>d4 d5 c4 c6 cxd5 e6 dxe6 fxe6 Nf3 Bb4+ Nc3 Ba5...</td>
      <td>D10</td>
      <td>Slav Defense: Exchange Variation</td>
      <td>5</td>
      <td>Slav Defense</td>
      <td>[d4, d5, c4, c6, cxd5]</td>
      <td>1345.5</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>l1NXvwaE</td>
      <td>True</td>
      <td>1.504130e+12</td>
      <td>1.504130e+12</td>
      <td>16</td>
      <td>resign</td>
      <td>black</td>
      <td>5+10</td>
      <td>a-00</td>
      <td>1322</td>
      <td>skinnerua</td>
      <td>1261</td>
      <td>d4 Nc6 e4 e5 f4 f6 dxe5 fxe5 fxe5 Nxe5 Qd4 Nc6...</td>
      <td>B00</td>
      <td>Nimzowitsch Defense: Kennedy Variation</td>
      <td>4</td>
      <td>Nimzowitsch Defense</td>
      <td>[d4, Nc6, e4, e5]</td>
      <td>1291.5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>mIICvQHh</td>
      <td>True</td>
      <td>1.504130e+12</td>
      <td>1.504130e+12</td>
      <td>61</td>
      <td>mate</td>
      <td>white</td>
      <td>5+10</td>
      <td>ischia</td>
      <td>1496</td>
      <td>a-00</td>
      <td>1500</td>
      <td>e4 e5 d3 d6 Be3 c6 Be2 b5 Nd2 a5 a4 c5 axb5 Nc...</td>
      <td>C20</td>
      <td>King's Pawn Game: Leonardis Variation</td>
      <td>3</td>
      <td>King's Pawn Game</td>
      <td>[e4, e5, d3]</td>
      <td>1498.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>kWKvrqYL</td>
      <td>True</td>
      <td>1.504110e+12</td>
      <td>1.504110e+12</td>
      <td>61</td>
      <td>mate</td>
      <td>white</td>
      <td>20+0</td>
      <td>daniamurashov</td>
      <td>1439</td>
      <td>adivanov2009</td>
      <td>1454</td>
      <td>d4 d5 Nf3 Bf5 Nc3 Nf6 Bf4 Ng4 e3 Nc6 Be2 Qd7 O...</td>
      <td>D02</td>
      <td>Queen's Pawn Game: Zukertort Variation</td>
      <td>3</td>
      <td>Queen's Pawn Game</td>
      <td>[d4, d5, Nf3]</td>
      <td>1446.5</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9tXo1AUZ</td>
      <td>True</td>
      <td>1.504030e+12</td>
      <td>1.504030e+12</td>
      <td>95</td>
      <td>mate</td>
      <td>white</td>
      <td>30+3</td>
      <td>nik221107</td>
      <td>1523</td>
      <td>adivanov2009</td>
      <td>1469</td>
      <td>e4 e5 Nf3 d6 d4 Nc6 d5 Nb4 a3 Na6 Nc3 Be7 b4 N...</td>
      <td>C41</td>
      <td>Philidor Defense</td>
      <td>5</td>
      <td>Philidor Defense</td>
      <td>[e4, e5, Nf3, d6, d4]</td>
      <td>1496.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>20053</th>
      <td>EfqH7VVH</td>
      <td>True</td>
      <td>1.499791e+12</td>
      <td>1.499791e+12</td>
      <td>24</td>
      <td>resign</td>
      <td>white</td>
      <td>10+10</td>
      <td>belcolt</td>
      <td>1691</td>
      <td>jamboger</td>
      <td>1220</td>
      <td>d4 f5 e3 e6 Nf3 Nf6 Nc3 b6 Be2 Bb7 O-O Be7 Ne5...</td>
      <td>A80</td>
      <td>Dutch Defense</td>
      <td>2</td>
      <td>Dutch Defense</td>
      <td>[d4, f5]</td>
      <td>1455.5</td>
      <td>3</td>
    </tr>
    <tr>
      <th>20054</th>
      <td>WSJDhbPl</td>
      <td>True</td>
      <td>1.499698e+12</td>
      <td>1.499699e+12</td>
      <td>82</td>
      <td>mate</td>
      <td>black</td>
      <td>10+0</td>
      <td>jamboger</td>
      <td>1233</td>
      <td>farrukhasomiddinov</td>
      <td>1196</td>
      <td>d4 d6 Bf4 e5 Bg3 Nf6 e3 exd4 exd4 d5 c3 Bd6 Bd...</td>
      <td>A41</td>
      <td>Queen's Pawn</td>
      <td>2</td>
      <td>Queen's Pawn</td>
      <td>[d4, d6]</td>
      <td>1214.5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>20055</th>
      <td>yrAas0Kj</td>
      <td>True</td>
      <td>1.499698e+12</td>
      <td>1.499698e+12</td>
      <td>35</td>
      <td>mate</td>
      <td>white</td>
      <td>10+0</td>
      <td>jamboger</td>
      <td>1219</td>
      <td>schaaksmurf3</td>
      <td>1286</td>
      <td>d4 d5 Bf4 Nc6 e3 Nf6 c3 e6 Nf3 Be7 Bd3 O-O Nbd...</td>
      <td>D00</td>
      <td>Queen's Pawn Game: Mason Attack</td>
      <td>3</td>
      <td>Queen's Pawn Game</td>
      <td>[d4, d5, Bf4]</td>
      <td>1252.5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>20056</th>
      <td>b0v4tRyF</td>
      <td>True</td>
      <td>1.499696e+12</td>
      <td>1.499697e+12</td>
      <td>109</td>
      <td>resign</td>
      <td>white</td>
      <td>10+0</td>
      <td>marcodisogno</td>
      <td>1360</td>
      <td>jamboger</td>
      <td>1227</td>
      <td>e4 d6 d4 Nf6 e5 dxe5 dxe5 Qxd1+ Kxd1 Nd5 c4 Nb...</td>
      <td>B07</td>
      <td>Pirc Defense</td>
      <td>4</td>
      <td>Pirc Defense</td>
      <td>[e4, d6, d4, Nf6]</td>
      <td>1293.5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>20057</th>
      <td>N8G2JHGG</td>
      <td>True</td>
      <td>1.499643e+12</td>
      <td>1.499644e+12</td>
      <td>78</td>
      <td>mate</td>
      <td>black</td>
      <td>10+0</td>
      <td>jamboger</td>
      <td>1235</td>
      <td>ffbob</td>
      <td>1339</td>
      <td>d4 d5 Bf4 Na6 e3 e6 c3 Nf6 Nf3 Bd7 Nbd2 b5 Bd3...</td>
      <td>D00</td>
      <td>Queen's Pawn Game: Mason Attack</td>
      <td>3</td>
      <td>Queen's Pawn Game</td>
      <td>[d4, d5, Bf4]</td>
      <td>1287.0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>20058 rows × 20 columns</p>
</div>




```

```
