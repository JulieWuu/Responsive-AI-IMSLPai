import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from google import genai
from google.genai import types

COLUMNS = ["piece", "url", "title", "composer", "catalogue", "key",
           "movements", "date", "period", "style", "instrumentation", "duration"]

SYSTEM_INSTRUCTION = (
    "You are IMSLPai, a classical music assistant. Be concise and helpful. "
    "Always use your tools to get facts about pieces (catalogue numbers, keys, durations, "
    "instrumentation, movements, dates, styles). Never invent database facts. "
    "Composer names in the database are formatted 'Surname, Forename', so search by surname. "
    "The database has no biographies or popularity data, so use your own knowledge for a composer's "
    "personal life and for which works are most renowned, then confirm those pieces exist with your tools.\n\n"
    "How to handle common questions:\n"
    "1. About a composer: call search_composer, then give basic information about their works and life.\n"
    "2. A composer's works for an instrument: call search_pieces with composer and instrument, and list a few.\n"
    "3. A specific piece: call get_piece_details and give the year composed, average duration, style, "
    "key and number of movements.\n"
    "4. Recommendations: use search_pieces, then pick the most renowned or best-suited to what the user wants.\n"
    "5. Comparing composers: call compare_composers and compare era, style and instrument proportions.\n"
    "6. If the user asks about all or many pieces at once, give only essential details per piece, ideally "
    "under 50 words each.\n"
    "7. For vague or descriptive requests that don't map to a field, try search_by_description."
    "8. Before answering, consider all the fields you get back (key, period, style, instrumentation, "
    "movements, dates, durations), not just the one the user mentioned, and call several tools if that "
    "gives a better answer."
)


class IMSLPai:
    def __init__(self, csv_path: str, api_key: str, system_instruction: str = SYSTEM_INSTRUCTION):
        self._load_data(csv_path)

        self.client = genai.Client(api_key=api_key)
        # Gemini calls these methods automatically when it decides it needs them
        self.chat_session = self.client.chats.create(
            model="gemini-2.5-flash-lite",
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(maximum_remote_calls=20),
                tools=[
                    self.search_composer,
                    self.search_pieces,
                    self.get_piece_details,
                    self.compare_composers,
                    self.search_by_description,
                ],
            ),
        )

    # ---------- data setup ----------
    def _load_data(self, csv_path: str):
        self.df = pd.read_csv(csv_path, header=0, names=COLUMNS).fillna("")
        self.docs = (
                self.df["title"] + " " + self.df["composer"] + " " + self.df["catalogue"] + " "
                + self.df["key"] + " " + self.df["period"] + " " + self.df["style"] + " "
                + self.df["instrumentation"] + " " + self.df["movements"] + " "
                + self.df["date"] + " " + self.df["duration"]
        )
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.docs)

    # ---------- helpers ----------
    def _composer_rows(self, composer: str) -> pd.DataFrame:
        return self.df[self.df["composer"].str.contains(composer, case=False, regex=False)]

    @staticmethod
    def _compact(row) -> dict:
        return {
            "title": row["title"], "composer": row["composer"], "catalogue": row["catalogue"],
            "key": row["key"], "instrumentation": row["instrumentation"],
            "duration": row["duration"], "url": row["url"],
        }

    @staticmethod
    def _full(row) -> dict:
        return {
            "title": row["title"], "composer": row["composer"], "catalogue": row["catalogue"],
            "key": row["key"], "movements": row["movements"], "year_composed": row["date"],
            "period": row["period"], "style": row["style"],
            "instrumentation": row["instrumentation"], "duration": row["duration"], "url": row["url"],
        }

    @staticmethod
    def _top_counts(series: pd.Series, n: int) -> dict:
        counts = series[series != ""].value_counts().head(n)
        return {k: int(v) for k, v in counts.items()}

    def _composer_summary(self, composer: str) -> dict:
        rows = self._composer_rows(composer)
        if rows.empty:
            return {"error": f"No composer matching '{composer}' found in the database."}
        instruments = (
            rows["instrumentation"].str.split(",").explode().str.strip().str.lower()
        )
        instruments = instruments[instruments != ""]
        share = (instruments.value_counts().head(6) / len(rows) * 100).round(1)
        return {
            "matched_names": sorted(rows["composer"].unique())[:5],
            "total_pieces": int(len(rows)),
            "period": self._top_counts(rows["period"], 2),
            "common_styles": self._top_counts(rows["style"], 3),
            "percent_of_pieces_featuring_instrument": {k: float(v) for k, v in share.items()},
        }

    # ---------- tools the AI can call ----------
    def search_composer(self, composer: str) -> dict:
        """Get an overview of a composer from the database: number of pieces, era, styles,
        instrument breakdown and a few sample pieces. Search by surname, e.g. 'Vivaldi'."""
        summary = self._composer_summary(composer)
        if "error" not in summary:
            summary["sample_pieces"] = self._composer_rows(composer)["title"].tolist()
        return summary

    def search_pieces(self, composer: str = "", instrument: str = "", key: str = "",
                      period: str = "", keyword: str = "") -> dict:
        """Search pieces by any combination of composer surname, instrument, key, period,
        and a keyword that is searched across EVERY field (title, catalogue, movements,
        date, style, duration...). Leave a filter empty to ignore it. Returns ALL matches
        with all their details."""
        mask = pd.Series(True, index=self.df.index)
        for col, value in [("composer", composer), ("instrumentation", instrument),
                           ("key", key), ("period", period)]:
            if value:
                mask &= self.df[col].str.contains(value, case=False, regex=False)
        if keyword:
            mask &= self.docs.str.contains(keyword, case=False, regex=False)
        matches = self.df[mask]
        return {"total_matches": int(len(matches)),
                "pieces": [self._full(r) for _, r in matches.iterrows()]}

    def get_piece_details(self, composer: str, piece: str) -> dict:
        """Get full details (year composed, duration, style, key, movements, instrumentation)
        for a specific piece. `piece` can be a title or catalogue number, e.g. 'RV 242'."""
        rows = self._composer_rows(composer) if composer else self.df
        mask = (rows["piece"].str.contains(piece, case=False, regex=False)
                | rows["catalogue"].str.contains(piece, case=False, regex=False))
        found = rows[mask]
        if found.empty:
            return {"error": f"No piece matching '{piece}' found."}
        return {"results": [self._full(r) for _, r in found.iterrows()]}

    def compare_composers(self, composers: list[str]) -> dict:
        """Compare several composers (by surname) on era, styles and instrument proportions."""
        return {name: self._composer_summary(name) for name in composers}

    def search_by_description(self, query: str) -> dict:
        """Fuzzy keyword search across all fields. Returns every piece with any match."""
        scores = linear_kernel(self.vectorizer.transform([query]), self.matrix).ravel()
        idx = [i for i in scores.argsort()[::-1] if scores[i] > 0]
        return {"pieces": [self._full(self.df.iloc[i]) for i in idx]}

    # ---------- chat loop ----------
    def chat(self):
        print("Hello! I am IMSLPai, your personal music AI. Ask me anything about the music in my database.")
        print("Type 'quit' to exit.")
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() == "quit":
                print("IMSLPai: Goodbye!")
                break
            response = self.chat_session.send_message(user_input)
            print(f"\nIMSLPai: {response.text}")


if __name__ == "__main__":
    bot = IMSLPai(
        csv_path="premium_data.csv",
        api_key="API_KEY",
    )
    bot.chat()
