import numpy as np
import matplotlib.pyplot as plt
import random
import plotly.express as px


class TimeSeriesAnomaly:
    def __init__(self, length, period, amplitude, anomaly_length, anomaly_value):
        self.length = length
        self.period = period
        self.amplitude = amplitude
        self.anomaly_length = anomaly_length
        self.anomaly_value = anomaly_value

    def generate_sequence(self):
        t = np.arange(0, self.length)
        y = self.amplitude * np.sin(2 * np.pi * t / self.period) + 5 + np.random.normal(0, 0.5, self.length)
        y = np.maximum(y, 0)  # Ensure each element of y is at least 0
        self.anomaly_value = random.randint(int(min(y)), int(max(y)))
        self.t = t
        self.y = y
        return t, y

    def generate_anomaly(self, start_location, end_location, anomaly_type='zero'):
        anomaly = np.zeros(self.anomaly_length)
        if anomaly_type == 'inject':
            for i in range(self.anomaly_length):
                anomaly[i] = np.sin(2 * np.pi * i / self.period) * self.anomaly_value
        elif anomaly_type == 'random':
            anomaly = np.random.normal(0, self.anomaly_value / 2, self.anomaly_length)
        elif anomaly_type == 'zero':
            anomaly = self.y[start_location:end_location]
            anomaly = -anomaly
        return anomaly

    def inject_anomaly(self, t, y):
        peak_locations = np.array([i for i in range(len(y)) if y[i] > np.mean(y)])
        trough_locations = np.array([i for i in range(len(y)) if y[i] <= np.mean(y)])
        if len(peak_locations) < 7 or len(trough_locations) < 7:
            return t, y, 0, 0
        peak_location = peak_locations[random.randint(0, len(peak_locations) - 1)]
        trough_location = trough_locations[random.randint(0, len(trough_locations) - 1)]
        if peak_location > trough_location:
            start_location = trough_location
            end_location = peak_location
        else:
            start_location = peak_location
            end_location = trough_location
        if end_location - start_location < self.anomaly_length:
            return t, y, start_location, end_location
        start_location = random.randint(start_location, end_location - self.anomaly_length)
        end_location = start_location + self.anomaly_length
        anomaly = self.generate_anomaly(start_location=start_location, end_location=end_location)
        y[start_location:end_location] += anomaly
        return t, y, start_location, end_location

    def plot_sequence(self, t, y, start_location=None, end_location=None):
        plt.plot(t, y, label='sequence')
        if start_location is not None and end_location is not None:
            plt.axvspan(t[start_location], t[end_location], color='red', alpha=0.3, label='anomaly')
        plt.legend(loc='upper right')
        plt.show()

    def plot_sequence_plotly(self, t, y, start_location=None, end_location=None):
        fig = px.line(x=t, y=y, labels={'x': 't', 'y': 'y'}, title='Anomaly Injection')
        if start_location is not None and end_location is not None:
            fig.add_shape(
                type='rect',
                x0=t[start_location], x1=t[end_location], y0=min(y), y1=max(y),
                yref='paper', xref='x',
                fillcolor='red', opacity=0.3, layer='below', line_width=0,
            )
            fig.update_layout(shapes=[
                dict(x0=t[start_location], x1=t[end_location], y0=0, y1=1, xref='x', yref='paper', fillcolor='red',
                     opacity=0.3, layer='below', line_width=0)])
            fig.update_layout(annotations=[dict(x=0.5, y=0.5, text='Anomaly', showarrow=False)])
            fig.update_layout(template='plotly_dark')
        fig.show()


if __name__ == '__main__':
    start_location, end_location = 0, 0
    length = 1000
    period = random.randint(50, 200)
    anomaly_length = random.randint(int(length / 100), int(length / 20))
    sequence = TimeSeriesAnomaly(length=length, period=period, amplitude=5, anomaly_length=anomaly_length,
                                 anomaly_value=30)
    t, y = sequence.generate_sequence()
    anomaly_value = random.randint(0, int(max(y)))
    t, y, start_location, end_location = sequence.inject_anomaly(t=t, y=y)
    sequence.plot_sequence_plotly(t=t, y=y, start_location=start_location, end_location=end_location)
    print()