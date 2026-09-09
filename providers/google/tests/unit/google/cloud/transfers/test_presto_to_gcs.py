#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from __future__ import annotations

from unittest import mock

import pytest

from airflow.providers.google.cloud.transfers.presto_to_gcs import (
    PrestoToGCSOperator,
    _PrestoToGCSPrestoCursorAdapter,
)

TASK_ID = "test-presto-to-gcs"
SQL = "SELECT * FROM memory.default.test_multiple_types"
BUCKET = "gs://test"
FILENAME = "test_{}.ndjson"

PRESTO_TO_GCS_PATH = "airflow.providers.google.cloud.transfers.presto_to_gcs"


@pytest.fixture
def cursor():
    return mock.MagicMock()


@pytest.fixture
def adapter(cursor):
    return _PrestoToGCSPrestoCursorAdapter(cursor)


class TestPrestoToGCSPrestoCursorAdapter:
    def test_delegates_rowcount(self, adapter, cursor):
        assert adapter.rowcount is cursor.rowcount

    def test_delegates_close(self, adapter, cursor):
        adapter.close()

        cursor.close.assert_called_once_with()

    def test_description_peeks_first_row_when_uninitialised(self, adapter, cursor):
        cursor.fetchone.return_value = ["A"]

        assert adapter.description is cursor.description
        cursor.fetchone.assert_called_once_with()
        assert adapter.rows == [["A"]]

    def test_description_does_not_peek_again_once_initialised(self, adapter, cursor):
        adapter.initialized = True

        assert adapter.description is cursor.description
        cursor.fetchone.assert_not_called()

    @pytest.mark.parametrize("method", ["execute", "executemany"])
    def test_reset_buffer_on_new_statement(self, adapter, cursor, method):
        adapter.initialized = True
        adapter.rows = [["stale"]]

        result = getattr(adapter, method)("SELECT 1")

        assert adapter.initialized is False
        assert adapter.rows == []
        assert result is getattr(cursor, method).return_value
        getattr(cursor, method).assert_called_once_with("SELECT 1")

    def test_peekone_does_not_consume_the_row(self, adapter, cursor):
        cursor.fetchone.return_value = ["A"]

        assert adapter.peekone() == ["A"]
        assert adapter.initialized is True
        assert adapter.fetchone() == ["A"]
        cursor.fetchone.assert_called_once_with()

    def test_fetchone_falls_through_to_cursor_when_buffer_empty(self, adapter, cursor):
        cursor.fetchone.return_value = ["A"]

        assert adapter.fetchone() == ["A"]

    def test_fetchmany_stops_at_end_of_result_set(self, adapter, cursor):
        cursor.fetchone.side_effect = [["A"], ["B"], None]

        assert adapter.fetchmany(5) == [["A"], ["B"]]

    def test_fetchmany_defaults_to_cursor_arraysize(self, adapter, cursor):
        cursor.arraysize = 2
        cursor.fetchone.side_effect = [["A"], ["B"], ["C"]]

        assert adapter.fetchmany() == [["A"], ["B"]]

    def test_iteration_yields_every_row_then_stops(self, adapter, cursor):
        cursor.fetchone.side_effect = [["A"], ["B"], None]

        assert iter(adapter) is adapter
        assert list(adapter) == [["A"], ["B"]]

    def test_next_raises_stop_iteration_when_exhausted(self, adapter, cursor):
        cursor.fetchone.return_value = None

        with pytest.raises(StopIteration):
            next(adapter)


class TestPrestoToGCSOperator:
    def test_init(self):
        op = PrestoToGCSOperator(
            task_id=TASK_ID, sql=SQL, bucket=BUCKET, filename=FILENAME, presto_conn_id="presto_custom"
        )

        assert op.sql == SQL
        assert op.bucket == BUCKET
        assert op.filename == FILENAME
        assert op.presto_conn_id == "presto_custom"

    def test_presto_conn_id_defaults_to_presto_default(self):
        op = PrestoToGCSOperator(task_id=TASK_ID, sql=SQL, bucket=BUCKET, filename=FILENAME)

        assert op.presto_conn_id == "presto_default"

    @pytest.mark.parametrize(
        ("field_type", "expected_type"),
        [
            pytest.param("BOOLEAN", "BOOL", id="mapped-type"),
            pytest.param("varchar", "STRING", id="lowercase-is-normalised"),
            pytest.param("DECIMAL(2, 10)", "NUMERIC", id="type-arguments-stripped"),
            pytest.param("TIME WITH TIME ZONE", "STRING", id="timezone-aware-time-degrades-to-string"),
            pytest.param("ARRAY(INTEGER)", "STRING", id="unmapped-type-falls-back-to-string"),
        ],
    )
    def test_field_to_bigquery(self, field_type, expected_type):
        op = PrestoToGCSOperator(task_id=TASK_ID, sql=SQL, bucket=BUCKET, filename=FILENAME)

        assert op.field_to_bigquery(("col", field_type)) == {"name": "col", "type": expected_type}

    def test_convert_type_passes_the_value_through(self):
        op = PrestoToGCSOperator(task_id=TASK_ID, sql=SQL, bucket=BUCKET, filename=FILENAME)

        value = object()

        assert op.convert_type(value, "DATE") is value

    @mock.patch(f"{PRESTO_TO_GCS_PATH}.PrestoHook")
    def test_query_wraps_the_cursor_in_the_adapter(self, mock_presto_hook):
        op = PrestoToGCSOperator(task_id=TASK_ID, sql=SQL, bucket=BUCKET, filename=FILENAME)

        result = op.query()

        mock_presto_hook.assert_called_once_with(presto_conn_id="presto_default")
        cursor = mock_presto_hook.return_value.get_conn.return_value.cursor.return_value
        cursor.execute.assert_called_once_with(SQL)
        assert isinstance(result, _PrestoToGCSPrestoCursorAdapter)
        assert result.cursor is cursor
