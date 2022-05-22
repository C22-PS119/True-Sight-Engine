import sys
import csv
import re
import html


class DataFilter:
    def __init__(self, html_decode=False, emoji_filter=False, unicode_filter=False, skip_empty=False, unescape=False) -> None:
        self.html_decode = html_decode
        self.emoji_filter = emoji_filter
        self.unicode_filter = unicode_filter
        self.skip_empty = skip_empty
        self.unescape = unescape

    def _html_filter(self, text: str):
        try:
            return html.unescape(text)
        except Exception as ex:
            print('err_HTML FILTER')
            raise Exception(ex)

    def _emoji_filter(self, text: str, replace_string=''):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   "]+", flags=re.UNICODE)
        try:
            return emoji_pattern.sub(replace_string, text)
        except Exception as ex:
            print('err_EMOJI FILTER')
            raise Exception(ex)

    def _unicode_filter(self, text: str, replace_string=''):
        try:
            return re.sub(r'[^\x00-\x7F]+', replace_string, str(text))
        except Exception as ex:
            print('err_UNICODE FILTER')
            raise Exception(ex)

    def _unescape(self, text: str):
        try:
            return text.encode('utf-8').decode('unicode_escape')
        except Exception as ex:
            print('err_UNESCAPE')
            raise Exception(ex)

    def stringFilter(self, text: str, replace_emoji='', replace_unicode=''):
        if text is None:
            return None
        if self.skip_empty and text == '':
            return None
        if self.html_decode:
            text = self._html_filter(text)
        if self.emoji_filter:
            text = self._emoji_filter(text, replace_emoji)
        if self.unicode_filter:
            text = self._unicode_filter(text, replace_unicode)
        if self.unescape:
            text = self._unescape(text)

        return text

    def arrayFilter(self, array, replace_emoji='', replace_unicode=''):
        if self.skip_empty:
            return [self.stringFilter(x, replace_emoji, replace_unicode) for x in array if not self.stringFilter(x, replace_emoji, replace_unicode) is None]
        else:
            return [self.stringFilter(x, replace_emoji, replace_unicode) for x in array]

    def dictionaryFilter(self, dict_array, lookup_header, replace_emoji, replace_unicode):
        filtered = list()
        for item in dict_array:
            item = dict(item)
            add_to_filtered = True
            for header in lookup_header:
                if item.get(header, '') == '' and self.skip_empty:
                    add_to_filtered = False
                    break
                else:
                    item[header] = self.stringFilter(
                        item.get(header, None), replace_emoji, replace_unicode)

            if add_to_filtered:
                filtered.append(item)

        return filtered


def main():
    if (len(sys.argv) > 2):
        encoding = 'utf-8'
        replace_emoji, replace_unicode = '', ' '
        output_file = sys.argv[2]
        for arg in sys.argv[2:]:
            if arg.startswith('--encoding'):
                encoding = arg.split('=')[1]
            if arg.startswith('--replace_emoji'):
                replace_emoji = arg.split('=')[1]
            if arg.startswith('--replace_unicode'):
                replace_unicode = arg.split('=')[1]
            if arg.startswith('-o'):
                output_file = arg.split('=')[1]

        source_file = open(sys.argv[1], 'r', encoding=encoding)
        dest_file = open(output_file, 'w', encoding=encoding, newline='')
        reader = csv.DictReader(source_file)
        writer = csv.DictWriter(dest_file, reader.fieldnames)

        data_filter = DataFilter(
            '--html' in sys.argv[2:],
            '--emoji' in sys.argv[2:],
            '--unicode' in sys.argv[2:],
            '--skip-empty' in sys.argv[2:],
            '--unescape' in sys.argv[2:]
        )
        data = list(reader)
        filtered = data_filter.dictionaryFilter(
            data, ['title', 'byline', 'content'], replace_emoji, replace_unicode)
        writer.writeheader()
        writer.writerows(filtered)
    elif (len(sys.argv) == 2):
        print("Please input destination file!")
    else:
        print("Please input source file!")


if __name__ == '__main__':
    main()
