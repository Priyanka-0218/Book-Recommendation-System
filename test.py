
import user_profiling
import os
import pandas as pd
import argparse

def get_recommendation(user_id=None,book_title=None,top_n=10):

    if user_id and book_title:
        print('cannot have both user_id and book_title')
        exit(1)
    profiling=user_profiling.UserProfiling()


    if not os.path.exists('profiles.pkl'):
        profiling.create_profiles()

    df=pd.read_csv('data/book_names_final.csv')


    if book_title.lower()=='yes':
        book_title_input=input('Enter Title of the book:\n')
        book_id=df[df.book_title==book_title_input].book_id
        if book_id.empty:
            print('Book not found')
        else:
            book_id=book_id.iloc[0]

    

    recommendations, scores = profiling.recommend_books(user_id,book_id,top_n=20)
    if recommendations is None:
        return print('no recommendations found')
    titles=[]
    
    for i in list(recommendations):
        titles.append(df[df.book_id==i].book_title.values[0])
    d=pd.DataFrame({'book_title':titles,'similarity_score':scores})
    print(d)

def main():
    parser = argparse.ArgumentParser(description="Get recommendations.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--user_id', type=int, help='User ID')
    group.add_argument('--book_title', type=str, help='Book title')
    parser.add_argument('--top_n', type=int, help='Number of recommendations', default=10)
    
    args = parser.parse_args()

    user_id = args.user_id
    book_title = args.book_title
    top_n = args.top_n
    
    get_recommendation(user_id=user_id, book_title=book_title, top_n=top_n)

if __name__ == "__main__":
    main()