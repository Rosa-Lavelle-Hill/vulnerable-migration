import pandas as pd
import datetime as dt


from Functions.Plotting import plot_hist
from Functions.Database import select
from Functions.Reporting import PrintDescriptives, CreateReport

start_time = dt.datetime.now()

f = open("SQL/Queries/Select/Senders_tab.txt")
send_query = f.read()

f = open("SQL/Queries/Select/Rec_tab.txt")
rec_query = f.read()

f = open("SQL/Queries/Select/Sender_Receivers.txt")
sender_rec_query = f.read()

if __name__ == '__main__':
    send_df = select(send_query)
    rec_df = select(rec_query)
    # sender_rec_df = select(sender_rec_query)


# print desciptives:
percentiles = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

save_path = "Results/Descriptives/Sender-Receivers/"
save_name = "Sender_receivers_descriptives.txt"
send_descript  = PrintDescriptives(percentiles=percentiles,
                  save_path=save_path,
                  df=send_df,
                  save_name= save_name,
                  cols=["transaction_amount",
                  "sender_night_tower_frequency",
                  "sender_pre_bal",
                  "receiver_pre_bal"],
                  print_title="Senders",
                  first=True
                  )

rec_descript = PrintDescriptives(percentiles=percentiles,
                  save_path=save_path,
                  df=rec_df,
                  save_name= save_name,
                  cols=["receiver_night_tower_frequency"],
                  print_title="Receivers",
                  first=False)


all_acc = pd.DataFrame(pd.concat([rec_df["receiver_msisdn_acc"],
                                   send_df["sender_msisdn"]],
                                   axis=0, ignore_index=True))
all_night_frequencies = pd.DataFrame(pd.concat([rec_df["receiver_night_tower_frequency"],
                                   send_df["sender_night_tower_frequency"]],
                                   axis=0, ignore_index=True))
all_night_frequencies = pd.DataFrame(pd.concat([all_acc,all_night_frequencies],
                                   axis=1, ignore_index=True))

all_night_frequencies.drop_duplicates(inplace=True)
all_night_frequencies.columns= ["id","all_night_frequencies"]


PrintDescriptives(percentiles=percentiles,
                  save_path=save_path,
                  df=all_night_frequencies,
                  save_name= save_name,
                  cols="all_night_frequencies",
                  print_title="All",
                  first=False)

processing_time = dt.datetime.now() - start_time
print("run time: " + str(processing_time), file=open(str(save_path) + save_name, "a"))

# Plot histograms:

fig_size = (4,4)
save_path = "Results/Descriptives/Sender-Receivers/Histograms/"

# plot night_tower_freq:

Senders_night_tower_freq = plot_hist(save_name="Senders_night_tower_freq",
                                     fig_size=fig_size,
                                     x=send_df["sender_night_tower_frequency"],
                                     bins=50,
                                     save_path=save_path,
                                     title="Sender night mode tower frequency")

plot_hist(save_name="Senders_night_tower_freq_large",
                                     fig_size=(5,5),
                                     x=send_df["sender_night_tower_frequency"],
                                     bins=50,
                                     save_path=save_path,
                                     title="Sender night mode tower frequency",
                                     xlim=(0,2000))

Rec_night_tower_freq = plot_hist(save_name="Rec_night_tower_freq",
                                     fig_size=fig_size,
                                     x=rec_df["receiver_night_tower_frequency"],
                                     bins=50,
                                     save_path=save_path,
                                     title="Receiver night mode tower frequency")

# Plot pre-bal:

Senders_pre_bal_freq = plot_hist(save_name="Senders_pre_bal_freq",
                                     fig_size=fig_size,
                                     x=send_df["sender_pre_bal"],
                                     bins=50,
                                     save_path=save_path,
                                     title="Sender pre-balance frequency")

Rec_pre_bal_freq = plot_hist(save_name="Rec_pre_bal_freq",
                                     fig_size=fig_size,
                                     x=send_df["receiver_pre_bal"],
                                     bins=50,
                                     save_path=save_path,
                                     title="Receiver pre-balance frequency")
# all frequencies:

plot_hist(save_name="All_night_tower_freq_large",
                                     fig_size=(5,5),
                                     x=all_night_frequencies["all_night_frequencies"],
                                     bins=50,
                                     save_path=save_path,
                                     title="All night mode tower frequency",
                                     xlim=(0,1200))

plot_hist(save_name="Transaction_amount",
                                     fig_size=(5,5),
                                     x=send_df["transaction_amount"],
                                     bins=50,
                                     save_path=save_path,
                                     title="Transaction amount"
                                     )

plot_hist(save_name="Transaction_amount_zoom",
                                     fig_size=(5,5),
                                     x=send_df["transaction_amount"],
                                     bins=50,
                                     save_path=save_path,
                                     title="Transaction amount",
                                     xlim = (0,151250)
                                     )

# Create report:
save_path = "Results/Descriptives/Sender-Receivers/"
save_name = "Sender_receivers_descriptives.txt"
f = open(save_path+save_name)
descript = f.read()

sender_html = send_descript.to_html()
rec_html = rec_descript.to_html()

report_dict = {"title": "Sender-Receivers",
                     "Senders": sender_html,
                     "Senders_night_tower_freq": Senders_night_tower_freq,
                     "Rec_night_tower_freq": Rec_night_tower_freq,
                     "Senders_pre_bal_freq": Senders_pre_bal_freq,
                     "Rec_pre_bal_freq": Rec_pre_bal_freq,
                     "Receivers": rec_html
                     }

if __name__ == "__main__":
    CreateReport(dict=report_dict,
                 pdf_save_name="pdf/Sender-Receiver.pdf",
                 template="html/sender_receiver.html")


print('done!')