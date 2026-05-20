class FuzzyFireSystem:
    def __init__(self):
        pass

    def _membership(self, x, a, b, c):
        """Triangle Membership Function"""
        if x <= a or x >= c:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a)
        elif b < x < c:
            return (c - x) / (c - b)
        return 0.0

    def _trapezoid(self, x, a, b, c, d):
        """Trapezoid Membership (Good for outer boundaries)"""
        if x <= a or x >= d:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a)
        elif b < x < c:
            return 1.0
        elif c < x < d:
            return (d - x) / (d - c)
        return 0.0

    def calculate_risk(self, temp, wind):
        """
        Input: Temp (F), Wind (mph)
        Output: Risk Percentage (0-100)
        """
        
        # 1. FUZZIFICATION
        # Temperature Membership
        temp_low = self._trapezoid(temp, -100, 0, 50, 70)
        temp_med = self._membership(temp, 60, 75, 90)
        temp_high = self._trapezoid(temp, 80, 95, 200, 300)

        # Wind Membership
        wind_low = self._trapezoid(wind, -10, 0, 5, 12)
        wind_med = self._membership(wind, 8, 15, 22)
        wind_high = self._trapezoid(wind, 18, 25, 100, 150)

        # 2. INFERENCE (Apply Rules)
        rules = []

        # Rule 1: High Temp + High Wind = CRITICAL
        rules.append((min(temp_high, wind_high), 100))
        # Rule 2: High Temp + Med Wind = HIGH
        rules.append((min(temp_high, wind_med), 80))
        # Rule 3: Med Temp + High Wind = HIGH
        rules.append((min(temp_med, wind_high), 75))
        # Rule 4: Med Temp + Med Wind = CAUTION
        rules.append((min(temp_med, wind_med), 50))
        # Rule 5: Low Temp = SAFE
        rules.append((temp_low, 10))

        # 3. DEFUZZIFICATION (Weighted Average)
        numerator = sum([w * score for w, score in rules])
        denominator = sum([w for w, score in rules])

        if denominator == 0:
            return 0.0
            
        final_score = numerator / denominator
        return final_score