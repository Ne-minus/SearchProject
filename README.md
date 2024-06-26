# SearchProject

## Строение кода  
В папке **isdb_nw** содержатся:   
1. [**\_\_main\_\_.py** ](https://github.com/hse-courses-tokubetsu/hw2-Ne-minus/blob/main/__main__.py)   -- основной файл для запуска  

   **Команда для запуска:**
   ```
   python __main__.py --help
   ```
   Так как в проекте реализовани интерфейс для командной строки, можно использовать команду ```--help``` с подробным описанием того, какие команды и синтаксис реализованы внутри проекта.
   
   **Существует 3 основных команды:**
  * ```download_models``` - скачивает необходимые модели и загружает их в оперативную память, запускается автоматически
  * ```search``` - реализует поиск, и если индекс еще не подсчитан, подсчет индекса. После первого вызова команды в директории проекта будет создана папка **data**, где будут находиться модели и подсчитанные индексы.
    команда принимает на вход 3 параметра:
      * аргумент QUERY -- запрос (в кавычках, если больше одного слова)
      * опция  INDEX -- имя индекса (по дефолту -- bm25, можно выбрать navec, w2v, bert)
      * опция NUMBER -- количество текстов в выдаче поисковика (по дефолту -- 5)
  * ```stats``` -- выдает табличку со временем и памятью, затраченными на поиск

2. [**loader.py**](https://github.com/hse-courses-tokubetsu/hw2-Ne-minus/blob/main/loader.py) -- содержит функции для загрузки моделей
3. [**index.py**](https://github.com/hse-courses-tokubetsu/hw2-Ne-minus/blob/main/index.py) -- содержит классы с реализацией индекса и поиска для разных метрик/моделей
4. [**stats.py**](https://github.com/hse-courses-tokubetsu/hw2-Ne-minus/blob/main/stats.py) -- подсчет времени и памяти, затраченных на выполнение поиска, результаты подсчета записаны в [stats.csv](https://github.com/hse-courses-tokubetsu/hw2-Ne-minus/blob/main/stats.csv)
5. [**preprocessing.py**](https://github.com/hse-courses-tokubetsu/hw2-Ne-minus/blob/main/preprocessing.py) -- код препроцессинга (очистка от знаков препинания, латиницы, цифр + приписывание частеречных тегов)
6. [**requirements.txt**](https://github.com/hse-courses-tokubetsu/hw2-Ne-minus/blob/main/requirements.txt) -- требующиеся библиотеки
7. [**isdb_hw2.csv**](https://github.com/hse-courses-tokubetsu/hw2-Ne-minus/blob/main/isdb_hw2.csv) -- корпус (содержит метаинформацию, сырой текст, лемматизированный текст и лемматизированный с частеречными тегами)
8. [**flask_app.py**](https://github.com/hse-courses-tokubetsu/project-Ne-minus/blob/main/main_files/flask_app.py) -- файл для запуска веб-приложения  
    **Команда для запуска:**
   
   ```
   python flask_app.py
   ```
После запуска необходимо перейти по ссылке в консоли - откроется сайт, нажав на кнопку ```Start searching!```, пользователь перейдет на страницу поиска, где необходимо заполнить все поля. После этого откроется страница с выдачей - тема, тэги, дата загрузки новости на сайт, название новости (кликабельное, по которому можно перейти в источник). Также поисковик поддерживает работу с невалидными запросами - выдает страницу с просьбой изменить запрос или метрику.  

9. [**Dockerfile**](https://github.com/hse-courses-tokubetsu/project-Ne-minus/blob/main/main_files/Dockerfile) -- файл с инструкциями по созданию контейнера с приложением
     
Во внешней папке содержится:  
1. [**docker-compose.yml**](https://github.com/hse-courses-tokubetsu/project-Ne-minus/blob/main/docker-compose.yml) -- файл с описанием необходимых докер-контейнеров (порты, название контейнеров, хоста и т.д.), нужный для их создания  
   В докере находится два контейнера: один - для самого приложения, второй - для базы данных.

##  Корпус  
Я взяла [датасет лента.ру](https://www.kaggle.com/datasets/yutkin/corpus-of-russian-news-articles-from-lenta), сделала стандартный препроцессинг (знаки препинания, цифры), посчитала длины всех вхождений, также убрала из корпуса всю латиницу, так как это новостной датасет и там много различных англоязычных слов и названий, которые загрязняют потом словарь. Новостной датасет показался мне интересным, так как на новостных сайтах часто встречаются статьи с клик-бейтом или иногда фейковыми новостями. Мне стало интересно, как различные поисковые метрики справятся с этими являениями. Из всего корпуса я создала политическо-экономичекий подкорпус размером в 2000 новостных статей.

*P.S. исходный датасет очень большой, поэтому здесь его не прикрепляю. Для запуска кода он не нужен, но если интересно его можно посмотреть по [ссылке](https://www.kaggle.com/datasets/yutkin/corpus-of-russian-news-articles-from-lenta).* 


## Модели

| Модель      | Источник                |
| :-------------: |:------------------:| 
| word2vec     | [ruwikiruscorpora_upos_cbow_300_10_2021](http://vectors.nlpl.eu/repository/20/220.zip)   |
| navec   | [navec_hudlit_v1_12B_500K_300d_100q.tar](https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar) |
| bert  | [ai-forever/sbert_large_nlu_ru](https://huggingface.co/ai-forever/sbert_large_nlu_ru)         |

