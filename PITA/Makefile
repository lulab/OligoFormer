FILES = \
	lib/add_column.pl \
	lib/body.pl \
	lib/cap.pl \
	lib/cut.pl \
	lib/extract_sequence.pl \
	lib/fasta2stab.pl \
	lib/find_potential_mirna_targets.pl \
	lib/format_number.pl \
	lib/join.pl \
	lib/libfile.pl \
	lib/libtable.pl \
	lib/load_args.pl \
	lib/merge_columns.pl \
	lib/modify_column.pl \
	lib/pita_run.pl \
	lib/RNAddG_compute.pl \
	lib/sequence_helpers.pl \
	lib/stab2length.pl \
	lib/tab2feature_gxt.pl \
	lib/transpose.pl \
	lib/uniquify.pl \
	lib/average_rows.pl \
	lib/libstats.pl \
	lib/ltqnorm.pl \
	pita_prediction.pl

CURR_DIR = $(shell pwd | sed 's/\//\\\//g')

install:
	$(foreach file, $(FILES), \
		mv $(file) tmp; \
		sed 's/EXE_BASE_DIR/$(CURR_DIR)/g' tmp > $(file); \
		rm -f tmp; \
	)
	chmod -R 755 lib Bin pita_prediction.pl; \
	chmod 775 known_mirs;
