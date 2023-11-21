## Notes for annotations

There are three types of annotations files in this folder, and they are generated for different purposes. 

- `instances_{subset}.json`
- `main_{subset}.json` 
- `index_{subset}.json`

The `instances_{subset}.json` annotations include all four types of images and seven types of layout elements. They are the complete version of the HJDataset and can be used for different tasks. The layout element bounding boxes (`annotation -> bbox`) are recorded using the XYWH format, following the Microsoft COCO style. Besides, you can use the page type annotation (`image -> category_id`) for page classification tasks. The corresponding information for `category_ids` also appears in categories with supercatergory being `page_type`. Additionally, the `parent_id` and `next_id` provide extra information for the text regions and can be used for hierarchical layout understanding and read order understanding tasks.

When adopting Deep Learning models to our dataset, we find it could be inconvenient to use the complete version. For example, you may only want to train models for the main pages rather than all images. To this end, we repack the annotations and create the main and index annotations for the two types of pages separately. In these files, only pages with the specific category appear, and categories do not contain the ids for `page_types`.