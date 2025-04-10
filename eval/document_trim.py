import re


# Helper function to count words in a text
def word_count(text):
    return len(re.findall(r'\w+', text))


# Define a simple Node class for the tree representation
class Node:
    def __init__(self, heading, level, content=""):
        self.heading = heading  # the heading text (including the '#' symbols)
        self.level = level  # heading level (e.g., 1,2,3)
        self.content = content  # text content under this heading (not in children)
        self.children = []  # list of child Node objects

    def total_text(self):
        """Return the full text for this node and all children."""
        texts = [self.heading] if self.heading else []
        if self.content:
            texts.append(self.content)
        for child in self.children:
            texts.append(child.total_text())
        return "\n".join(texts)

    def total_word_count(self):
        return word_count(self.total_text())


# Parse the article into a tree
def parse_article(text):
    # regex to capture headings. E.g., line starting with one or more '#'
    heading_pattern = re.compile(r'^(#{1,6})\s*(.*)$')
    lines = text.splitlines()
    root = Node(heading="", level=0)
    stack = [root]

    # For each line, check if it's a heading
    for line in lines:
        m = heading_pattern.match(line)
        if m:
            level = len(m.group(1))
            heading_text = line.strip()
            new_node = Node(heading=heading_text, level=level)
            # find the parent (node with level less than current)
            while stack and stack[-1].level >= level:
                stack.pop()
            stack[-1].children.append(new_node)
            stack.append(new_node)
        else:
            # Append text to the current node's content
            if stack:
                # maintain paragraph separation by adding a space
                stack[-1].content += " " + line.strip()
            else:
                root.content += " " + line.strip()
    return root


# Function to find all nodes at a given maximum depth that have non-empty content.
def find_deepest_nodes(node):
    # If a node has children, we want to get nodes from the deepest level.
    if not node.children:
        return [node] if node.content.strip() else []
    else:
        results = []
        for child in node.children:
            results.extend(find_deepest_nodes(child))
        return results


# Function to trim the article by removing content from the smallest (shortest) section at the deepest level.
def trim_article(root, target_word_count=2000):
    # While the total word count is above the target
    while root.total_word_count() > target_word_count:
        # Gather all candidate nodes from the deepest level (lowest level headings)
        candidates = find_deepest_nodes(root)
        # Exclude the root node
        candidates = [node for node in candidates if node.level > 0]
        if not candidates:
            break  # nothing to remove

        # Sort candidates by their word count (lowest first)
        candidates.sort(key=lambda n: word_count(n.content))
        # Remove content from the smallest candidate
        smallest = candidates[0]
        print(f"Removing content from section: {smallest.heading} ({word_count(smallest.content)} words)")
        smallest.content = ""  # remove all content in this node

        # Optionally, if a node has no content and no children, remove it from parent's children
        remove_empty_nodes(root)
    return root


def remove_empty_nodes(node):
    # Recursively remove child nodes that have no content and no children
    new_children = []
    for child in node.children:
        remove_empty_nodes(child)
        if child.content.strip() or child.children:
            new_children.append(child)
    node.children = new_children


# Function to reassemble the article text from the tree
def assemble_article(node):
    texts = []
    if node.heading:
        texts.append(node.heading)
    if node.content.strip():
        texts.append(node.content.strip())
    for child in node.children:
        texts.append(assemble_article(child))
    return "\n".join(texts)

# Example usage:
if __name__ == '__main__':
    # Read your article from a file or string
    with open('../data/storm_interface/65th_Annual_Grammy_Awards.txt', 'r', encoding='utf-8') as f:
        article_text = f.read()

    root = parse_article(article_text)
    print("Original word count:", root.total_word_count())

    trimmed_root = trim_article(root, target_word_count=2000)
    trimmed_text = assemble_article(trimmed_root)
    print("Trimmed word count:", word_count(trimmed_text))

    print(trimmed_text)
