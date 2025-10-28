import csv


class LinkRecorder:
    def __init__(self):
        self.link_utilization = {}

    def record_utilization(self, time, link_type, utilization):
        if time not in self.link_utilization:
            self.link_utilization[time] = {}
        self.link_utilization[time][link_type] = utilization

    def generate_report(self, output_dir, rate, time):
        """모든 플로우에 대한 최종 통계 보고서를 CSV 파일로 저장합니다."""
        if not self.link_utilization:
            print("No flow statistics to report.")
            return

        all_stats = []
        for time_stamp, utilization_by_link_type in self.link_utilization.items():
            stats = {}
            stats['time'] = time_stamp
            for link_type, utilization in utilization_by_link_type.items():
                stats[link_type] = utilization
            all_stats.append(stats)

        if not all_stats:
            print("No data with link utilization to report.")
            return

        filepath = f"{output_dir}/link_stats_{rate}_{time}.csv"

        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                headers = all_stats[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writeheader()
                writer.writerows(all_stats)

            print(f"\n--- Flow Statistics Summary ---")
            print(f"Flow statistics report saved to '{filepath}'")
            print("-----------------------------")
        except (IOError, IndexError) as e:
            print(f"Error writing to CSV file: {e}")