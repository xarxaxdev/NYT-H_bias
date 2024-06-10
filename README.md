# NYT-H_bias

```
head -n 99999 generated_models/*/metrics.csv > all.csv

````





### Data Example
```python

# We want to predict relation
# Using sentence + word1 + type1 + word2 + type2
{
    "instance_id": "NONNADEV#193662",
    "bag_id": "NONNADEV#91512",
    "relation": "/people/person/place_lived",
    "bag_label": "unk", # `unk` means the bag is not annotated, otherwise `yes` or `no`.
    "sentence": "Instead , he 's the kind of writer who can stare at the wall of his house in New Rochelle -LRB- as he did with '' Ragtime '' -RRB- , think about the year the house was built (1906) , follow his thoughts to the local tracks that once brought trolleys from New Rochelle to New York City and wind up with a book featuring Theodore Roosevelt , Scott Joplin , Emma Goldman , Stanford White and Harry Houdini . ''",
    "head": {
        "guid": "/guid/9202a8c04000641f8000000000176dc3",
        "word": "Stanford White",
        "type": "/influence/influence_node,/people/deceased_person" # type for entities, split by comma if one entity has many types
    },
    "tail": {
        "guid": "/guid/9202a8c04000641f80000000002f8906",
        "word": "New York City",
        "type": "/architecture/architectural_structure_owner,/location/citytown"
    }
}
```
