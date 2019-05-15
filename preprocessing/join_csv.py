import pandas as pd

qns_df = pd.read_csv("Questions.csv", encoding="ISO-8859-1")
tags_df = pd.read_csv("Tags.csv", encoding="ISO-8859-1", dtype={'Tag': str})
ans_df = pd.read_csv("Answers.csv", encoding="ISO-8859-1")

ans_df['Body'] = ans_df['Body'].astype(str)

# Group answers by ParentId and concatenate them
print("Grouping answers by ParentId...")
grouped_ans = ans_df.groupby("ParentId")['Body'].apply(lambda ans: ' '.join(ans))
grouped_ans.reset_index()

grouped_ans_final = pd.DataFrame({'Id':grouped_ans.index, 'Answers':grouped_ans.values})

qns_df.drop(columns=['OwnerUserId', 'CreationDate', 'ClosedDate'], inplace=True)

# Merge answers with questions based on Id role
print("Joining answers to questions...")
new_df = qns_df.merge(grouped_ans_final, on='Id')

tags_df['Tag'] = tags_df['Tag'].astype(str)

# Group tags by Id and concatenate them
print("Grouping tags by Id...")
grouped_tags = tags_df.groupby("Id")['Tag'].apply(lambda tags: ' '.join(tags))
grouped_tags.reset_index()

grouped_tags_final = pd.DataFrame({'Id':grouped_tags.index, 'Tags':grouped_tags.values})

# Merge tags with main DataFrame
print("Joining tags to questions and answers...")
new_df = new_df.merge(grouped_tags_final, on='Id')

# Save as new csv file
new_df.to_csv('joined_data.csv', index=False)