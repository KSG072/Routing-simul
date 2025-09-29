import csv
from collections import deque

class FlowStats:
    """단일 플로우의 통계를 추적하는 데이터 클래스입니다."""
    def __init__(self, fkey):
        self.fkey = fkey
        self.status = 'off'  # 플로우 상태 ('on' 또는 'off')
        self.on_timing = deque()
        self.off_timing = deque()
        self.total_packets_generated = 0
        self.succeeded_packets = 0
        self.failed_packets = 0
        self.failed_on_ISL = 0
        self.failed_on_before_GSL = 0
        self.failed_on_after_GSL = 0
        self.total_queueing_delay = 0.0
        self.total_propagation_delay = 0.0
        self.total_e2e_delay = 0.0
        self.total_hops = 0

    def on(self, time):
        self.status = 'on'
        self.on_timing.append(time)

    def off(self, time):
        self.status = 'off'
        self.off_timing.append(time)

    def record_outcome(self, packet):
        """패킷 처리 결과를 통계에 반영합니다."""
        if packet.success:
            self.succeeded_packets += 1
            self.total_queueing_delay += sum(packet.queuing_delays)
            self.total_propagation_delay += packet.propagation_delays
            self.total_e2e_delay += sum(packet.queuing_delays) + packet.propagation_delays + packet.transmission_delay
            self.total_hops += len(packet.result)-1
        else:
            self.failed_packets += 1

    def get_stats(self):
        """계산된 최종 통계를 딕셔너리 형태로 반환합니다."""
        # 객체의 모든 속성을 복사
        stats = self.__dict__.copy()

        # 'status' 속성 제거
        del stats['status']

        # 통계치 계산
        succeeded = self.succeeded_packets
        total_processed = self.total_packets_generated

        # 계산된 통계치를 딕셔너리에 추가
        stats['avg_e2e_delay_ms'] = self.total_e2e_delay / succeeded if succeeded > 0 else 0
        stats['avg_queueing_delay_ms'] = self.total_queueing_delay / succeeded if succeeded > 0 else 0
        stats['avg_propagation_delay_ms'] = self.total_propagation_delay / succeeded if succeeded > 0 else 0
        stats['avg_hops'] = self.total_hops / succeeded if succeeded > 0 else 0
        stats['drop_rate'] = self.failed_packets / total_processed if total_processed > 0 else 0

        # deque를 list로 변환
        stats['on_timing'] = list(self.on_timing)
        stats['off_timing'] = list(self.off_timing)

        return stats

class FlowRecorder:
    """모든 플로우의 통계를 기록하고 관리합니다."""
    def __init__(self):
        self.flows = {}
        self.on_flows = set()

    def is_new_flow(self, fkey):
        return fkey not in self.flows.keys()

    def create_flow(self, fkey):
        if fkey in self.flows.keys():
            print(f"{fkey} is already created")
        self.flows[fkey] = FlowStats(fkey)

    def get_flow(self, fkey):
        return self.flows[fkey]

    def record_flow_on(self, fkey, time, num_packets):
        """플로우 시작 및 패킷 생성을 기록합니다."""
        if fkey not in self.on_flows:
            flow_stat = self.get_flow(fkey)
            self.on_flows.add(flow_stat)
            if flow_stat.status == 'off':
                flow_stat.on(time)
            flow_stat.total_packets_generated += num_packets

    def record_flow_end(self, time):
        """플로우 종료 시간을 기록합니다."""
        for key, stat in self.flows.items():
            if key not in self.on_flows:
                if stat.status == 'on':
                    stat.off(time)
        self.on_flows.clear()

    def record_packet_outcome(self, packet):
        """처리 완료된 패킷의 결과를 해당 플로우 통계에 기록합니다."""
        fkey = (packet.source, packet.destination)
        if fkey in self.flows:
            self.flows[fkey].record_outcome(packet)

    def generate_report(self, output_dir, rate):
        """모든 플로우에 대한 최종 통계 보고서를 CSV 파일로 저장합니다."""
        if not self.flows:
            print("No flow statistics to report.")
            return

        all_stats = []
        for fkey, flow_stat in sorted(self.flows.items()):
            stats = flow_stat.get_stats()
            if stats['total_packets_generated'] > 0:
                # CSV 호환성을 위해 튜플과 리스트를 문자열로 변환
                stats['fkey'] = str(stats['fkey'])
                stats['on_timing'] = str(stats['on_timing'])
                stats['off_timing'] = str(stats['off_timing'])
                all_stats.append(stats)

        if not all_stats:
            print("No flows with generated packets to report.")
            return

        filepath = f"{output_dir}/flow_stats_{rate}.csv"

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
